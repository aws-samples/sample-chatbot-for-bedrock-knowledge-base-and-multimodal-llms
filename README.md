# Multimodal Chatbot with Bedrock Knowledge Base Integration

This project is a Streamlit application that combines the capabilities of Bedrock knowledge bases with multimodal language models (LLMs). The application allows users to ask questions and receive relevant responses based on the selected knowledge base and multimodal LLM model. Users can also upload images to supplement their queries.

## Installation

1. Clone the repository:
```
git clone https://github.com/aws-samples/rag-multimodal.git
cd rag-multimodal/app
```


2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Overwrite the `config.json` file with the your preferred parameters

4. If you want to create a new knoweldge base, take a look at the CLI tool `create_kb.py` in the `scripts/` folder
```
python scripts/create_kb.py --help 
```
In that case, you would need to have an AWS Role with the follwoing Policies:
    * IAMFullAccess
    * AWSLambda_FullAccess
    * AmazonS3FullAccess
    * AmazonBedrockFullAccess
    * Custom policy for Amazon OpenSearch Serverless such as
    ```
    {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "aoss:*",
                        "Resource": "*"
                    }
                ]
            }
    ```

## Usage

1. Start the Streamlit application:

```
streamlit run main.py

```

2. The application will open in your default web browser.
3. Use the sidebar to select the desired Bedrock model, knowledge base, and toggle streaming mode.
4. Enter your query in the chat input box and press Enter.
5. The application will retrieve relevant documents from the selected knowledge base and provide a response from the chosen multimodal LLM.
6. If streaming mode is enabled, the response will be displayed incrementally as it is generated.
7. You can upload images to supplement your query by clicking the "Choose one or more images" button in the sidebar.
8. To start a new chat, click the "New Chat" button in the sidebar.

