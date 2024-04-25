import json
import boto3
import streamlit as st
from utils.bedrock import BedrockHandler, KBHandler

# Clear Chat History function
def clear_screen():
    st.session_state.messages = [
        {"role": "assistant", "content": configs["start_message"]}
    ]
    bedrock_handler.messages = []


def get_all_kbs(all_kb: dict) -> dict:
    result = {}
    for kb in all_kb["knowledgeBaseSummaries"]:
        result[kb["name"]] = kb["knowledgeBaseId"]
    return result

with open('config.json') as f:
    configs = json.load(f)


# Page title
st.set_page_config(page_title=configs["page_title"])

claude_models = {
    "Anthropic Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Anthropic Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Anthropic Claude 3 Opus (Currently unavailable)": "anthropic.claude-3-opus-20240229-v1:0"
}
bedrock_agents_client = boto3.client(service_name="bedrock-agent", region_name=configs["bedrock_region"])
all_kbs = get_all_kbs(bedrock_agents_client.list_knowledge_bases(maxResults=10))



with st.sidebar:
    st.title(configs["page_title"])
    streaming_on = st.toggle("Streaming", value=True)
    uploaded_files = st.file_uploader("Choose one or more images", accept_multiple_files=True, type=["png", "jpg"])
    bedrock_model = st.selectbox("Choose Bedrock model", claude_models.keys(), index=1)
    knoweldge_base_selection = st.selectbox("Choose a Knoweldge base", ["None"] + list(all_kbs.keys()), index=1)
    st.button("New Chat", on_click=clear_screen, type="primary")



bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=configs["bedrock_region"])
model_id = claude_models[bedrock_model]

bedrock_handler = BedrockHandler(bedrock_runtime, model_id, configs["claude_model_params"])

selected_kb = all_kbs[knoweldge_base_selection] if knoweldge_base_selection != "None" else None
retriever = KBHandler(bedrock_runtime, configs["bedrock_region"], configs["kb_configs"], kb_id=selected_kb)

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
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    docs = retriever.get_relevant_docs(prompt)
    context = retriever.parse_kb_output_to_string(docs)
    st.session_state.bedrock_messages.append(bedrock_handler.user_message(prompt, context, uploaded_pics=uploaded_files))

    if streaming_on:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            stream = bedrock_handler.invoke_model_with_stream(st.session_state.bedrock_messages).get("body")
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_message = json.loads(chunk.get("bytes").decode())
                    full_response += BedrockHandler.get_body_from_stream_chunks(chunk_message)
                    placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            with st.expander("Show source details >"):
                st.write(retriever.parse_kb_output_to_reference(docs))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.bedrock_messages.append(bedrock_handler.assistant_message(full_response))
    else:
        with st.chat_message("assistant"):
            response = bedrock_handler.invoke_model(st.session_state.bedrock_messages)
            response_body = json.loads(response.get('body').read())
            response_body = response_body['content'][0]['text']
            st.write(response_body)
            with st.expander("Show source details >"):
                st.write(retriever.parse_kb_output_to_reference(docs))
            st.session_state.messages.append({"role": "assistant", "content": response_body})
            st.session_state.bedrock_messages.append(bedrock_handler.assistant_message(response_body))
    
    uploaded_files = []
