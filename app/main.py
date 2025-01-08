import json
import boto3
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, Union
from utils.bedrock import BedrockHandler, KBHandler, S3Handler
import base64
import time

def load_config():
    path = Path(__file__).parent.absolute()
    with open(path / "config.json", encoding="utf-8") as f:
        return json.load(f)

configs = load_config()

def clear_screen() -> None:
    """Clear the chat history and reset the messages."""
    st.session_state.messages = [
        {"role": "assistant", "content": configs["start_message"]}
    ]
    st.session_state.bedrock_messages = []
    if "video_job" in st.session_state:
        st.session_state.video_job = None

def get_all_kbs(all_kb: Dict[str, Any]) -> Dict[str, str]:
    """Extract knowledge base names and IDs from the response."""
    return {
        kb["name"]: kb["knowledgeBaseId"] 
        for kb in all_kb["knowledgeBaseSummaries"]
    }

def on_region_change() -> None:
    """Start new chat and refresh knowledge bases when region changes."""
    clear_screen()
    st.session_state.all_kbs = get_all_kbs(
        boto3.client(
            service_name="bedrock-agent", 
            region_name=configs["regions"][st.session_state.selected_region]
        ).list_knowledge_bases(maxResults=10)
    )

def get_video_status(client: Any, invocation_arn: str) -> Dict[str, Any]:
    """Get the status of a video generation job."""
    try:
        response = client.get_async_invoke(invocationArn=invocation_arn)
        return {
            "status": response.get("status", "Unknown"),
            "completed": response.get("status") == "Completed",
            "failed": response.get("status") == "Failed",
            "error": response.get("failureMessage", None)
        }
    except Exception as e:
        return {
            "status": "Error",
            "completed": False,
            "failed": True,
            "error": str(e)
        }

def setup_sidebar(configs: Dict[str, Any]) -> tuple[str, str, bool, str, str]:
    """Setup and handle sidebar UI elements."""
    st.sidebar.title(configs["page_title"])
    
    selected_region = st.sidebar.selectbox(
        "Choose Region",
        configs["regions"],
        index=0,
        on_change=on_region_change,
        key="selected_region"
    )
    
    available_models = list(configs["multimodal_llms"][selected_region].keys())
    selected_model = st.sidebar.selectbox(
        "Choose Bedrock model", available_models, index=1
    )
    
    is_image_model = "Nova Canvas" in selected_model
    is_video_model = "Nova Reel" in selected_model
    
    streaming_on = (
        st.sidebar.toggle("Streaming", value=True)
        if not (is_image_model or is_video_model)
        else False
    )
    
    kb_selection = (
        st.sidebar.selectbox(
            "Choose a Knowledge base",
            ["None"] + list(st.session_state.all_kbs.keys()),
            index=0
        )
        if not (is_image_model or is_video_model)
        else "None"
    )
    
    s3_uri = None
    if is_video_model:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        default_bucket = f"bedrock-video-generation-us-east-1-{account_id}"
        bucket_path = st.sidebar.text_input(
            "S3 Output Location",
            value=default_bucket
        )
        s3_uri = f"s3://{bucket_path or default_bucket}"
    
    if is_video_model:
        st.session_state.uploaded_files = st.sidebar.file_uploader("Supported file types are .png, .jpeg", accept_multiple_files=True)
    elif is_image_model:
        st.session_state.uploaded_files = st.sidebar.file_uploader("Supported file types are .png, .jpeg", accept_multiple_files=True)
    else:
        st.session_state.uploaded_files = st.sidebar.file_uploader("Upload a file", accept_multiple_files=True)
    st.sidebar.button("New Chat", on_click=clear_screen, type="primary")
    
    return selected_region, selected_model, streaming_on, kb_selection, s3_uri

def main():
    """Main application logic."""
    st.set_page_config(page_title=configs["page_title"])
    
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = "Frankfurt"
        
    if 'all_kbs' not in st.session_state:
        bedrock_agents_client = boto3.client(
            service_name="bedrock-agent",
            region_name=configs["regions"][st.session_state.selected_region]
        )
        st.session_state.all_kbs = get_all_kbs(
            bedrock_agents_client.list_knowledge_bases(maxResults=10)
        )

    selected_region, selected_model, streaming_on, kb_selection, s3_uri = setup_sidebar(configs)


    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=configs["regions"][selected_region]
    )
    
    model_id = configs["multimodal_llms"][selected_region][selected_model]
    model_params = {
        "nova-canvas": configs["nova_canvas_params"],
        "nova-reel": configs["nova_reel_params"],
        "anthropic": configs["claude_model_params"],
        "nova": configs["nova_model_params"]
    }
    
    params = next(
        (params for key, params in model_params.items() if key in model_id),
        configs["nova_model_params"]
    )
    
    bedrock_handler = BedrockHandler(bedrock_runtime, model_id, params)

    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=configs["regions"][selected_region]
    )
    
    selected_kb = (
        st.session_state.all_kbs[kb_selection]
        if kb_selection != "None"
        else None
    )
    
    retriever = KBHandler(
        bedrock_agent_runtime_client,
        configs["kb_configs"],
        kb_id=selected_kb
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": configs["start_message"]}
        ]

    if "bedrock_messages" not in st.session_state:
        st.session_state.bedrock_messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                if "text" in message["content"]:
                    st.write(message["content"]["text"])
                if "image" in message["content"]:
                    image_data = base64.b64decode(message["content"]["image"])
                    st.image(image_data)
            else:
                st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if "nova-reel" in model_id and not s3_uri:
            st.error("Please provide an S3 output location for video generation")
            return

        docs = (
            retriever.get_relevant_docs(prompt)
            if not ("nova-canvas" in model_id or "nova-reel" in model_id)
            else []
        )
        context = retriever.parse_kb_output_to_string(docs) if docs else None
        
        user_msg = bedrock_handler.user_message(
            prompt,
            context,
            st.session_state.uploaded_files
        )
        
        if "nova-reel" in model_id:
            user_msg["s3_uri"] = s3_uri
            
        st.session_state.bedrock_messages.append(user_msg)
        
        with st.chat_message("assistant"):
            if "nova-canvas" in model_id:
                handle_image_generation(bedrock_handler, st.session_state.bedrock_messages)
            elif "nova-reel" in model_id:
                handle_video_generation(
                    bedrock_handler,
                    bedrock_runtime,
                    prompt,
                    s3_uri,
                    st.session_state.uploaded_files[0] if st.session_state.uploaded_files else None
                )
            else:
                handle_text_generation(
                    bedrock_handler,
                    st.session_state.bedrock_messages,
                    streaming_on,
                    docs,
                    retriever
                )

def handle_image_generation(bedrock_handler: BedrockHandler, messages: list) -> None:
    """Handle image generation and display."""
    image_data = bedrock_handler.invoke_model(messages)
    st.image(image_data, caption="Generated Image")
    full_response = {
        "text": "I've generated the image based on your prompt.",
        "image": base64.b64encode(image_data).decode('utf-8')
    }
    update_chat_history(full_response)

def handle_video_generation(
    bedrock_handler: BedrockHandler,
    bedrock_runtime: Any,
    prompt: str,
    s3_uri: str,
    uploaded_file: Optional[Any] = None
) -> None:
    """Handle video generation and status monitoring."""
    status_placeholder = st.empty()
    
    if "video_job" not in st.session_state:
        st.session_state.video_job = None
        
    if st.session_state.video_job is None:
        status_placeholder.info("Starting video generation. This may take a few minutes...")
        
        image_data = None
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_format = Path(uploaded_file.name).suffix[1:]
            image_data = (image_bytes, image_format)
            
        job_details = bedrock_handler.generate_video(prompt, s3_uri, image_data)
        st.session_state.video_job = job_details
        
        
    while True:
        status = get_video_status(bedrock_runtime, st.session_state.video_job["invocation_arn"])
        
        if status["completed"]:
            s3_details = st.session_state.video_job["s3_details"]
            video_exists, video_path = S3Handler().check_video_exists(
                s3_details["bucket"],
                s3_details["prefix"]
            )
            
            if video_exists:
                message = f"✅ Video generation completed! Video available at: s3://{s3_details['bucket']}/{video_path}"
                status_placeholder.success(message)
                update_chat_history({"text": message})
                st.session_state.video_job = None
                break
            status_placeholder.info("Video processing... Waiting for S3 upload to complete...")
                
        elif status["failed"]:
            message = f"❌ Video generation failed: {status['error']}"
            status_placeholder.error(message)
            update_chat_history({"text": message})
            st.session_state.video_job = None
            break
            
        else:
            status_placeholder.info(f"⏳ Generating Video... This can take up to 5 minutes. Status: {status['status']}")
            
        time.sleep(10)

def handle_text_generation(
    bedrock_handler: BedrockHandler,
    messages: list,
    streaming: bool,
    docs: list,
    retriever: KBHandler
) -> None:
    """Handle text generation with or without streaming."""
    if streaming:
        placeholder = st.empty()
        streamed_response = ""
        stream = bedrock_handler.invoke_model_with_stream(messages).get("stream")
        
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    streamed_response += event["contentBlockDelta"]["delta"]["text"]
                placeholder.markdown(streamed_response)
        full_response = {"text": streamed_response}
    else:
        response = bedrock_handler.invoke_model(messages)
        full_response = response["output"]["message"]["content"][0]["text"]
        st.write(full_response)

    if docs:
        with st.expander("Show source details >"):
            st.write(retriever.parse_kb_output_to_reference(docs))
    
    update_chat_history(full_response)

def update_chat_history(response: Union[str, Dict[str, Any]]) -> None:
    """Update chat history with new response."""
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    st.session_state.bedrock_messages.append(
        BedrockHandler.assistant_message(
            response["text"] if isinstance(response, dict) else response
        )
    )

if __name__ == "__main__":
    main()