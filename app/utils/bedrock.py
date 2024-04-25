from typing import Optional
import base64
import json


class BedrockHandler:
    def __init__(self, client, model_id, params):
        self.params = params
        self.model_id = model_id
        self.client = client

    @staticmethod
    def assistant_message(message: str) -> dict:
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        }   
        
    @staticmethod
    def user_message(message: str, context: Optional[str], uploaded_pics: Optional[list]) -> dict:
        context_message = f"You are a helful support engineer, answer the following question based on the provided context: \n\n {context} \n\n " if context else ""
        new_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{context_message} question: {message}"
                    }
                ]
            }
        if uploaded_pics:
            for uploaded_pic in uploaded_pics:
                bytes_data = uploaded_pic.read()
                img_base64 = base64.b64encode(bytes_data).decode("utf-8")
                new_message["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64 
                        }
                    }
                )
        return new_message

    def invoke_model(self, messages):
        body = json.dumps(self.params | {"messages": messages})
        return self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
    
    def invoke_model_with_stream(self, messages):
        body = json.dumps(self.params | {"messages": messages})
        return self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
    
    @staticmethod
    def get_body_from_stream_chunks(chunk: dict) -> str:
        if "delta" in chunk and "text" in chunk["delta"]:
            return chunk["delta"]["text"]
        return ""


class KBHandler:
    def __init__(self, client, kb_params: dict, kb_id: Optional[str] = None) -> None:
        self.client = client
        self.kb_id = kb_id
        self.params = kb_params

    def get_relevant_docs(self, prompt: str) -> list[dict]:
        return self.client.retrieve(
                retrievalQuery= {
                    'text': prompt
                },
                knowledgeBaseId=self.kb_id,
                retrievalConfiguration= self.params
        )["retrievalResults"] if self.kb_id else []

    @staticmethod
    def parse_kb_output_to_string(docs: list[dict]) -> str:
        return "\n\n".join(f"Document {i + 1}: {doc['content']['text']}" for i, doc in enumerate(docs))

    @staticmethod
    def parse_kb_output_to_reference(docs: list[dict]) -> dict:
        return {
            f"Document {i + 1}": {
                "text": doc["content"]["text"],
                "metadata": doc["location"],
                "score": doc["score"]
            } for i, doc in enumerate(docs)
        }
