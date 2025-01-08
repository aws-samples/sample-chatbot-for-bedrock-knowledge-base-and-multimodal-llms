import base64
import json
from typing import Optional, Union, Dict, List, Any
from pathlib import Path
import boto3
import streamlit as st

class S3Handler:
    """Handles S3-related operations for the application."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.client = boto3.client('s3', region_name=region_name)
    
    def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Create S3 bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
        except self.client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    return True
                except Exception as create_error:
                    st.error(f"Failed to create bucket: {str(create_error)}")
                    return False
            st.error(f"Error accessing bucket: {str(e)}")
            return False
    
    def check_video_exists(self, bucket: str, prefix: str) -> tuple[bool, str]:
        """Check if video file exists in specified S3 location."""
        try:
            prefix = f"{prefix}/" if not prefix.endswith('/') else prefix
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            if 'Contents' not in response:
                return False, ""
                
            for obj in response['Contents']:
                if obj['Key'].endswith('.mp4'):
                    return True, obj['Key']
            
            return False, ""
        except Exception as e:
            st.error(f"Error checking S3: {str(e)}")
            return False, ""

class BedrockHandler:
    """Handles interactions with Bedrock models and manages messages."""

    def __init__(self, client: Any, model_id: str, params: Dict[str, Any]):
        self.params = params
        self.model_id = model_id
        self.client = client
        self.is_image_model = "nova-canvas" in model_id
        self.is_video_model = "nova-reel" in model_id
        self.s3_handler = S3Handler()

    @staticmethod
    def assistant_message(message: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Create a message dictionary for assistant's response."""
        content = [{"text": message}]
        if image_data:
            content.append({"image": image_data})
        return {"role": "assistant", "content": content}

    @staticmethod
    def user_message(
        message: str,
        context: Optional[str] = None,
        uploaded_files: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Create a message dictionary for user's query."""
        context_message = (
            f"You are a helpful assistant, answer the following question based on the provided context: \n\n {context} \n\n "
            if context else ""
        )
        new_message = {
            "role": "user",
            "content": [{"text": f"{context_message} question: {message}"}],
        }

        if uploaded_files:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                extension = Path(uploaded_file.name).suffix[1:]
                
                if extension in ["png", "jpeg", "gif", "webp"]:
                    new_message["content"].append({
                        "image": {
                            "format": extension,
                            "source": {"bytes": bytes_data}
                        }
                    })
                elif extension in ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]:
                    new_message["content"].append({
                        "document": {
                            "format": extension,
                            "name": "random-doc-name",
                            "source": {"bytes": bytes_data}
                        }
                    })
        return new_message

    def generate_image(self, messages: List[Dict[str, Any]]) -> bytes:
        """Generate an image using Nova Canvas."""
        last_message = messages[-1]
        text_prompt = last_message["content"][0]["text"]

        body = {
            "taskType": "IMAGE_VARIATION" if len(last_message["content"]) > 1 else "TEXT_IMAGE",
            "imageGenerationConfig": {
                "numberOfImages": self.params.get("numberOfImages", 1),
                "height": self.params.get("height", 512),
                "width": self.params.get("width", 512),
                "cfgScale": self.params.get("cfgScale", 8.0)
            }
        }
        
        if len(last_message["content"]) > 1:
            images = []
            for content in last_message["content"]:
                if "image" in content:
                    image_format = content["image"].get("format", "png").lower().replace(".", "")
                    if image_format in["png", "jpeg"]:
                        image_bytes = (content["image"]["source"]["bytes"] 
                                    if isinstance(content["image"], dict) else content["image"])
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        images.append(base64_image)
                    else:
                        raise ValueError("Image format must be PNG or JPEG")
                    
            
            body["imageVariationParams"] = {
                "text": text_prompt,
                "negativeText": "bad quality, low resolution",
                "images": images,
                "similarityStrength": 0.7,
            }
        else:
            body["textToImageParams"] = {
                "text": text_prompt,
                "negativeText": "bad quality, low resolution"
            }
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        if "error" in response_body:
            raise Exception(f"Image generation error: {response_body['error']}")
        
        return base64.b64decode(response_body['images'][0])

    def generate_video(self, prompt: str, s3_uri: str, uploaded_image: Optional[tuple[bytes, str]] = None) -> Dict[str, Any]:
        """Generate a video using Nova Reel."""
        bucket = s3_uri.split("//")[1].split("/")[0]
        
        if not self.s3_handler.ensure_bucket_exists(bucket):
            raise Exception(f"Failed to create/verify S3 bucket: {bucket}")

        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": prompt
            },
            "videoGenerationConfig": {
                "durationSeconds": self.params.get("durationSeconds", 6),
                "fps": self.params.get("fps", 24),
                "dimension": self.params.get("dimension", "1280x720"),
            }
        }

        if uploaded_image:
            image_bytes, image_format = uploaded_image
            image_format = image_format.lower().replace(".", "")
            if image_format not in ["png", "jpeg"]:
                raise ValueError("Image format must be PNG or JPEG")

            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            model_input["textToVideoParams"]["images"] = [
                {
                    "format": image_format,
                    "source": {
                        "bytes": base64_image
                    }
                }
            ]

        response = self.client.start_async_invoke(
            modelId=self.model_id,
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": s3_uri}}
        )

        invocation_id = response["invocationArn"].split('/')[-1]
        return {
            "invocation_arn": response["invocationArn"],
            "s3_details": {
                "bucket": bucket,
                "prefix": invocation_id
            }
        }
    def invoke_model(self, messages: List[Dict[str, Any]]) -> Union[Dict[str, Any], bytes]:
        """Invoke the appropriate model based on type."""
        try:
            if self.is_image_model:
                return self.generate_image(messages)
            elif self.is_video_model:
                return self.generate_video(
                    messages[-1]["content"][0]["text"],
                    messages[-1].get("s3_uri")
                )
            else:
                return self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig={"temperature": 0.0},
                    additionalModelRequestFields={"top_k": 100} if "anthropic" in self.model_id else {}           
                )
        except Exception as e:
            st.error(f"Error invoking model: {str(e)}")
            return None

    def invoke_model_with_stream(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Invoke model with streaming."""
        if self.is_image_model or self.is_video_model:
            raise ValueError("Streaming is not supported for image or video generation models")
        return self.client.converse_stream(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={"temperature": 0.0},
            additionalModelRequestFields={"top_k": 100} if "anthropic" in self.model_id else {}
        )

class KBHandler:
    """Handles interactions with Bedrock knowledge bases."""

    def __init__(self, client: Any, kb_params: Dict[str, Any], kb_id: Optional[str] = None):
        self.client = client
        self.kb_id = kb_id
        self.params = kb_params

    def get_relevant_docs(self, prompt: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the knowledge base."""
        if not self.kb_id:
            return []
        
        return self.client.retrieve(
            retrievalQuery={"text": prompt},
            knowledgeBaseId=self.kb_id,
            retrievalConfiguration=self.params,
        )["retrievalResults"]

    @staticmethod
    def parse_kb_output_to_string(docs: List[Dict[str, Any]]) -> str:
        """Parse documents into a string format."""
        return "\n\n".join(
            f"Document {i + 1}: {doc['content']['text']}" 
            for i, doc in enumerate(docs)
        )

    @staticmethod
    def parse_kb_output_to_reference(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse documents into a reference format with metadata."""
        return {
            f"Document {i + 1}": {
                "text": doc["content"]["text"],
                "metadata": doc["location"],
                "score": doc["score"],
            }
            for i, doc in enumerate(docs)
        }