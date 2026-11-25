import json

def transcript(role: str, text: str) -> str:
    return json.dumps({"type": "transcript", "role": role, "text": text})

def interrupted() -> str:
    return json.dumps({"type": "interrupted"})

def error(message: str) -> str:
    return json.dumps({"type": "error", "message": message})