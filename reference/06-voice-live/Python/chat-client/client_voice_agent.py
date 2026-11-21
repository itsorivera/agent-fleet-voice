#!/usr/bin/env python3
"""voice_client.py - Cliente speech-to-speech para un agente de voz de Foundry (kind="voice")."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    RealtimeServerEventError,
    RealtimeServerEventResponseAudioDelta,
    RealtimeServerEventResponseAudioTranscriptDone,
    RealtimeServerEventSessionCreated,
)
from azure.identity import AzureCliCredential


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

DEFAULT_AGENT = "chat-agent"
SAMPLE_RATE = 24_000  # PCM16 mono
CHUNK_FRAMES = 2_400  # 100 ms


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_environment() -> None:
    for candidate in (REPO_ROOT / ".env", SCRIPT_DIR / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def resolve_project_endpoint() -> str:
    endpoint = os.environ.get("PROJECT_ENDPOINT") or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    if endpoint:
        return endpoint

    voice_live_endpoint = os.environ.get("AZURE_VOICELIVE_ENDPOINT")
    project_name = os.environ.get("AZURE_VOICELIVE_PROJECT_NAME")
    if voice_live_endpoint and project_name:
        return f"{voice_live_endpoint.rstrip('/')}/api/projects/{project_name}"

    raise SystemExit(
        "Falta PROJECT_ENDPOINT en el .env de la raíz "
        "(o AZURE_VOICELIVE_ENDPOINT + AZURE_VOICELIVE_PROJECT_NAME)."
    )


def stream_microphone(conn, stop: threading.Event) -> None:
    """Captura el micrófono y envía PCM16 al agente hasta que `stop` se active."""
    mic_q: queue.Queue[bytes] = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        mic_q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=CHUNK_FRAMES, callback=on_audio,
    ):
        while not stop.is_set():
            try:
                chunk = mic_q.get(timeout=0.2)
            except queue.Empty:
                continue
            conn.input_audio_buffer.append(audio=chunk)


def main() -> None:
    configure_console()
    load_environment()

    parser = argparse.ArgumentParser(description="Cliente speech-to-speech para un Foundry Voice Agent.")
    parser.add_argument("--agent", default=os.environ.get("AZURE_VOICELIVE_AGENT_ID", DEFAULT_AGENT))
    args = parser.parse_args()

    endpoint = resolve_project_endpoint()

    print("=" * 60)
    print("🎙️  FOUNDRY VOICE AGENT - SPEECH TO SPEECH")
    print("=" * 60)
    print(f"Proyecto: {endpoint}")
    print(f"Agente  : {args.agent}")

    with AIProjectClient(endpoint=endpoint, credential=AzureCliCredential()) as project:
        agent = project.agents.get(agent_name=args.agent)
        definition = agent.versions.latest.definition

        if getattr(definition, "kind", None) != "voice":
            raise SystemExit(
                f"'{args.agent}' tiene kind={getattr(definition, 'kind', None)!r}. "
                "Este cliente solo sirve para agentes kind='voice'."
            )

        stop = threading.Event()

        with project.voice_agents.realtime.connect(agent_name=args.agent) as conn, \
             sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as speaker:

            threading.Thread(target=stream_microphone, args=(conn, stop), daemon=True).start()
            print("\n🎧 Habla cuando quieras (Ctrl+C para salir).\n")

            try:
                while True:
                    event = conn.recv(timeout=None)

                    if isinstance(event, RealtimeServerEventSessionCreated):
                        print(f"🔗 Conversación: {event.conversation_id}")

                    elif isinstance(event, RealtimeServerEventResponseAudioDelta):
                        speaker.write(event.delta)

                    elif isinstance(event, RealtimeServerEventResponseAudioTranscriptDone):
                        print(f"🤖 Agente: {event.transcript}")

                    elif isinstance(event, RealtimeServerEventError):
                        raise RuntimeError(f"Error del servicio: {event.error.message}")

                    elif getattr(event, "type", None) == "input_audio_buffer.speech_started":
                        # Barge-in: el usuario interrumpe, se descarta el audio pendiente.
                        speaker.abort()
                        speaker.start()
            except KeyboardInterrupt:
                print("\n👋 Adiós.")
            finally:
                stop.set()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)
