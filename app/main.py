import json
import boto3
import streamlit as st
from pathlib import Path
from utils.bedrock import BedrockHandler, KBHandler


# Clear Chat History function
def clear_screen() -> None:
    """Clear the chat history and reset the messages."""
    st.session_state.messages = [
        {"role": "assistant", "content": configs["start_message"]}
    ]
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


if __name__ == "__main__":
    path = Path(__file__).parent.absolute() # gets path of parent directory
    with open(path / "config.json", encoding="utf-8") as f:
        configs = json.load(f)

    # Page title
    st.set_page_config(page_title=configs["page_title"])
    bedrock_agents_client = boto3.client(
        service_name="bedrock-agent", region_name=configs["bedrock_region"]
    )
    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime", region_name=configs["bedrock_region"]
    )
    all_kbs = get_all_kbs(bedrock_agents_client.list_knowledge_bases(maxResults=10))

    with st.sidebar:
        st.title(configs["page_title"])
        streaming_on = st.toggle("Streaming", value=True)
        uploaded_files = st.file_uploader(
            "Choose one or more images", accept_multiple_files=True, type=["png", "jpg"]
        )
        selected_bedrock_model = st.selectbox(
            "Choose Bedrock model", configs["multimodal_llms"].keys(), index=1
        )
        knoweldge_base_selection = st.selectbox(
            "Choose a Knoweldge base", ["None"] + list(all_kbs.keys()), index=0
        )
        st.button("New Chat", on_click=clear_screen, type="primary")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=configs["bedrock_region"]
    )
    model_id = configs["multimodal_llms"][selected_bedrock_model]

    bedrock_handler = BedrockHandler(
        bedrock_runtime, model_id, configs["claude_model_params"]
    )

    selected_kb = (
        all_kbs[knoweldge_base_selection]
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
            st.write(message["content"])

    # Chat Input - User Prompt
    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        docs = retriever.get_relevant_docs(prompt)
        context = retriever.parse_kb_output_to_string(docs)
        st.session_state.bedrock_messages.append(
            bedrock_handler.user_message(prompt, context, uploaded_pics=uploaded_files)
        )
        full_response = ""
        if streaming_on:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                stream = bedrock_handler.invoke_model_with_stream(
                    st.session_state.bedrock_messages
                ).get("body")
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_message = json.loads(chunk.get("bytes").decode())
                        full_response += BedrockHandler.get_body_from_stream_chunks(
                            chunk_message
                        )
                        placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                with st.expander("Show source details >"):
                    st.write(retriever.parse_kb_output_to_reference(docs))
        else:
            with st.chat_message("assistant"):
                response = bedrock_handler.invoke_model(
                    st.session_state.bedrock_messages
                )
                response_body = json.loads(response.get("body").read())
                full_response = response_body["content"][0]["text"]
                st.write(full_response)
                with st.expander("Show source details >"):
                    st.write(retriever.parse_kb_output_to_reference(docs))

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.bedrock_messages.append(
            bedrock_handler.assistant_message(full_response)
        )
