import json
import boto3
import streamlit as st
from pathlib import Path
from utils.bedrock import BedrockHandler, KBHandler
import base64

# Clear Chat History function
def clear_screen() -> None:
    """Clear the chat history and reset the messages."""
    st.session_state.messages = [
        {"role": "assistant", "content": configs["start_message"]}
    ]
    st.session_state.bedrock_messages = []
    bedrock_handler.messages = []


def get_all_kbs(all_kb: dict) -> dict[str, str]:
    """
    Extract knowledge base names and IDs from the response.

    Args:
        all_kb (dict): The response from the list_knowledge_bases API call.

    Returns:
        Dict[str, str]: A dictionary mapping knowledge base names to their IDs.
    """
    result = {}
    for kb in all_kb["knowledgeBaseSummaries"]:
        result[kb["name"]] = kb["knowledgeBaseId"]
    return result

def on_region_change():
    """Start new chat and refresh knowledge bases when region changes""" 
    clear_screen()
    st.session_state.all_kbs = get_all_kbs(
        boto3.client(
            service_name="bedrock-agent", 
            region_name=configs["regions"][st.session_state.selected_region]
        ).list_knowledge_bases(maxResults=10)
    )

if __name__ == "__main__":
    path = Path(__file__).parent.absolute()  # gets path of parent directory
    with open(path / "config.json", encoding="utf-8") as f:
        configs = json.load(f)

    # Page title
    st.set_page_config(page_title=configs["page_title"])
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = "Frankfurt"
    bedrock_agents_client = boto3.client(
        service_name="bedrock-agent", region_name=configs["regions"][st.session_state.selected_region]
    )
    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime", region_name=configs["regions"][st.session_state.selected_region]
    )
    if 'all_kbs' not in st.session_state:
        st.session_state.all_kbs = get_all_kbs(
        bedrock_agents_client.list_knowledge_bases(maxResults=10)
    )
    
    with st.sidebar:
        st.title(configs["page_title"])
        selected_region = st.selectbox(
            "Choose Region",
            configs["regions"],
            index=0,
            on_change=on_region_change,
            key="selected_region"
        )
        availble_models = list(configs["multimodal_llms"][selected_region].keys())
        selected_bedrock_model = st.selectbox(
            "Choose Bedrock model", availble_models, index=1
        )
        
        # Only show streaming toggle for non-image/video models
        is_image_model = "Nova Canvas" in selected_bedrock_model
        is_video_model = "Nova Reel" in selected_bedrock_model
        
        if not (is_image_model or is_video_model):
            streaming_on = st.toggle("Streaming", value=True)
            knoweldge_base_selection = st.selectbox("Choose a Knoweldge base", ["None"] + list(st.session_state.all_kbs.keys()), index=0)
        else:
            streaming_on = False
            knoweldge_base_selection = "None"
            
        # Show S3 bucket input for Nova Reel
        s3_uri = None
        if is_video_model:
            s3_uri = st.text_input("S3 Output Location (s3://bucket-name/prefix/)")
            
        if not (is_image_model or is_video_model):
            uploaded_files = st.file_uploader(
                "Choose one or more images", accept_multiple_files=True
            )
        else:
            uploaded_files = None
            

        
        st.button("New Chat", on_click=clear_screen, type="primary")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=configs["regions"][selected_region]
    )
    model_id = configs["multimodal_llms"][selected_region][selected_bedrock_model]
    
    if "nova-canvas" in model_id:
        model_params = configs["nova_canvas_params"]
    elif "nova-reel" in model_id:
        model_params = configs["nova_reel_params"]
    elif "anthropic" in model_id:
        model_params = configs["claude_model_params"]
    else:
        model_params = configs["nova_model_params"]
    
    bedrock_handler = BedrockHandler(
        bedrock_runtime, model_id, model_params
    )

    selected_kb = (
        st.session_state.all_kbs[knoweldge_base_selection]
        if knoweldge_base_selection != "None"
        else None
    )
    retriever = KBHandler(
        bedrock_agent_runtime_client, configs["kb_configs"], kb_id=selected_kb
    )

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": configs["start_message"]}
        ]

    if "bedrock_messages" not in st.session_state.keys():
        st.session_state.bedrock_messages = []

    # Display or clear chat messages
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

    # Chat Input - User Prompt
    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if is_video_model and not s3_uri:
            st.error("Please provide an S3 output location for video generation")
        else:
            docs = retriever.get_relevant_docs(prompt) if not (is_image_model or is_video_model) else []
            context = retriever.parse_kb_output_to_string(docs) if docs else None
            
            user_msg = bedrock_handler.user_message(prompt, context, uploaded_files)
            if is_video_model:
                user_msg["s3_uri"] = s3_uri
                
            st.session_state.bedrock_messages.append(user_msg)
            
            with st.chat_message("assistant"):
                if is_image_model:
                    image_data = bedrock_handler.invoke_model(st.session_state.bedrock_messages)
                    st.image(image_data, caption="Generated Image")
                    full_response = {
                        "text": "I've generated the image based on your prompt.",
                        "image": base64.b64encode(image_data).decode('utf-8')
                    }
                elif is_video_model:
                    full_response = {
                        "text": f"Video generation job submitted. This can take a couple of minutes. Plese check your S3 location: {s3_uri} for the result."
                    }
                    st.write(full_response["text"])
                    bedrock_handler.invoke_model(st.session_state.bedrock_messages)
                elif streaming_on:
                    placeholder = st.empty()
                    streamed_response = ""
                    stream = bedrock_handler.invoke_model_with_stream(
                        st.session_state.bedrock_messages
                        ).get("stream")
                    if stream:
                        for event in stream:
                            if "contentBlockDelta" in event:
                                streamed_response += event["contentBlockDelta"]["delta"]["text"]
                            placeholder.markdown(streamed_response)
                            placeholder.markdown(streamed_response)
                    full_response = {"text": streamed_response}
                else:
                    response = bedrock_handler.invoke_model(
                        st.session_state.bedrock_messages
                    )
                    full_response = response["output"]["message"]["content"][0]["text"]
                    st.write(full_response)

                if docs:
                    with st.expander("Show source details >"):
                        st.write(retriever.parse_kb_output_to_reference(docs))

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            st.session_state.bedrock_messages.append(
                bedrock_handler.assistant_message(full_response["text"])
            )
