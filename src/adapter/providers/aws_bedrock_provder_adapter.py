import os
import boto3
from langchain_aws import ChatBedrockConverse
from src.core.ports.llm_provider_port import LLMProviderPort

class AWSBedrockProviderAdapter(LLMProviderPort):
    def __init__(self):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        self._session = boto3.Session(
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )
        self._bedrock_client = self._session.client("bedrock-runtime")

    def get_llm(self, llm_id: str) -> ChatBedrockConverse:
        return ChatBedrockConverse(
            client=self._bedrock_client,
            model_id=llm_id
        )
