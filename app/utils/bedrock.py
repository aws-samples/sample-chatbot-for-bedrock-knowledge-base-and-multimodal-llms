"""
Handles communication to Bedrock and KnowledgeBases
"""

import base64
import json
from typing import Optional
from pathlib import Path


class BedrockHandler:
    """
    A class to handle interactions with Bedrock models and manage messages.
    """

    def __init__(self, client, model_id: str, params: dict):
        """
        Initialize the BedrockHandler with a client, model ID, and parameters.

        Args:
            client: The Bedrock client object.
            model_id (str): The ID of the Bedrock model to use.
            params (dict): The parameters for the model.
        """
        self.params = params
        self.model_id = model_id
        self.client = client

    @staticmethod
    def assistant_message(message: str) -> dict:
        """
        Create a message dictionary representing an assistant's response.

        Args:
            message (str): The text content of the assistant's response.

        Returns:
            dict: A message dictionary with the role set to "assistant" and the content set to the provided message.
        """
        return {"role": "assistant", "content": [{"text": message}]}

    @staticmethod
    def user_message(
        message: str,
        context: Optional[str] = None,
        uploaded_files: Optional[list] = None,
    ) -> dict:
        """
        Create a message dictionary representing a user's query, optionally including context and uploaded images.

        Args:
            message (str): The text content of the user's query.
            context (str, optional): The context information to include in the message. Defaults to None.
            uploaded_pics (list, optional): A list of uploaded image files. Defaults to None.

        Returns:
            dict: A message dictionary with the role set to "user" and the content containing the provided message,
                  context (if available), and base64-encoded image data (if provided).
        """
        context_message = (
            f"You are a helpful assistant, answer the following question based on the provided context: \n\n {context} \n\n "
            if context
            else ""
        )
        new_message = {
            "role": "user",
            "content": [{"text": f"{context_message} question: {message}"}],
        }
        if uploaded_files:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                extension = Path(uploaded_file.name).suffix[1:]
                print(extension)
                if extension in ["png", "jpeg", "gif", "webp"]:
                    new_message["content"].append(
                        {
                            "image": {
                                "format": extension,
                                "source": {
                                    "bytes": bytes_data,
                                },
                            }
                        }
                    )
                elif extension in ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt","md"]:
                    new_message["content"].append(
                        {
                            "document": {
                                "format": extension,
                                "name": "random-doc-name",
                                "source": {
                                    "bytes": bytes_data,
                                },
                            }
                        }
                    )
        return new_message

    def invoke_model(self, messages: list) -> dict:
        """
        Invoke the Bedrock model with the provided messages and return the response.

        Args:
            messages (list): A list of message dictionaries containing the conversation history.

        Returns:
            dict: The response from the Bedrock model.
        """
        return self.client.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={"temperature": 0.0},
            additionalModelRequestFields={"top_k": 100},
        )

    def invoke_model_with_stream(self, messages: list) -> dict:
        """
        Invoke the Bedrock model with the provided messages and return a streaming response.

        Args:
            messages (list): A list of message dictionaries containing the conversation history.

        Returns:
            dict: The streaming response from the Bedrock model.
        """
        return self.client.converse_stream(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={"temperature": 0.0},
            additionalModelRequestFields={"top_k": 100},
        )


class KBHandler:
    """
    A class to handle interactions with Bedrock knowledge bases and retrieve relevant documents.
    """

    def __init__(self, client, kb_params: dict, kb_id: Optional[str] = None):
        """
        Initialize the KBHandler with a client, knowledge base parameters, and an optional knowledge base ID.

        Args:
            client: The Bedrock client object.
            kb_params (dict): The parameters for the knowledge base.
            kb_id (str, optional): The ID of the knowledge base to use. Defaults to None.
        """
        self.client = client
        self.kb_id = kb_id
        self.params = kb_params

    def get_relevant_docs(self, prompt: str) -> list[dict]:
        """
        Retrieve relevant documents from the knowledge base based on the provided prompt.

        Args:
            prompt (str): The prompt or query to search for relevant documents.

        Returns:
            list[dict]: A list of dictionaries representing the retrieved documents.
        """
        return (
            self.client.retrieve(
                retrievalQuery={"text": prompt},
                knowledgeBaseId=self.kb_id,
                retrievalConfiguration=self.params,
            )["retrievalResults"]
            if self.kb_id
            else []
        )

    @staticmethod
    def parse_kb_output_to_string(docs: list[dict]) -> str:
        """
        Parse the retrieved documents into a string format.

        Args:
            docs (list[dict]): A list of dictionaries representing the retrieved documents.

        Returns:
            str: A string containing the content of the retrieved documents, separated by newlines.
        """
        return "\n\n".join(
            f"Document {i + 1}: {doc['content']['text']}" for i, doc in enumerate(docs)
        )

    @staticmethod
    def parse_kb_output_to_reference(docs: list[dict]) -> dict:
        """
        Parse the retrieved documents into a dictionary format with metadata.

        Args:
            docs (list[dict]): A list of dictionaries representing the retrieved documents.

        Returns:
            dict: A dictionary mapping document numbers to dictionaries containing the document text, metadata, and score.
        """
        return {
            f"Document {i + 1}": {
                "text": doc["content"]["text"],
                "metadata": doc["location"],
                "score": doc["score"],
            }
            for i, doc in enumerate(docs)
        }
