"""
This class is mostly copied from: https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/02_KnowledgeBases_and_RAG/0_create_ingest_documents_test_kb.ipynb
Note: you need to have the following polices in order to be able to activate this code:
* IAMFullAccess
* AWSLambda_FullAccess
* AmazonS3FullAccess
* AmazonBedrockFullAccess
* Custom policy for Amazon OpenSearch Serverless such as:
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
"""
import json
import os
import boto3
import sys
import argparse
from botocore.exceptions import ClientError
import pprint
import random
from retrying import retry
from opensearchpy import RequestError
sys.path.append('..')  # Add the parent directory to the Python path


parser = argparse.ArgumentParser(description='Create and ingest documents into a knowledge base')
parser.add_argument('--region_name', type=str, required=False, help='AWS region name', default="us-west-2")
parser.add_argument('--knowledge_base_name', type=str, required=True, help='Knowledge base name')
parser.add_argument('--bucket_name', type=str, required=False, help='S3 Bucket name that should save the data')
parser.add_argument('--copy_from_local', required=False, help='If set, files are copied from `data` directory to the S3 bucket', default=True)
parser.add_argument('--vectorstore_name', type=str, required=False, help='Name of the vector store', default="bedrock-sample-rag-vs")
parser.add_argument('--index_name', type=str, required=False, help='Name of the opensearch index', default="bedrock-sample-rag-index")

args = parser.parse_args()

region_name = args.region_name
if region_name not in ["us-east-1", "us-west-2"]:
    raise Exception("The region for needs to be sit to us-east-1 or us-west-2")

os.environ["AWS_DEFAULT_REGION"] = region_name
suffix = random.randrange(200, 900)
boto3.setup_default_session(region_name=region_name)
sts_client = boto3.client('sts')
boto3_session = boto3.session.Session()
bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region_name)
service = 'aoss'
s3_client = boto3.client('s3')
account_id = sts_client.get_caller_identity()["Account"]
s3_suffix = f"{region_name}-{account_id}"
bucket_name = f'bedrock-kb-{s3_suffix}' if not args.bucket_name else args.bucket_name
pp = pprint.PrettyPrinter(indent=2)

try:
    s3_client.head_bucket(Bucket=bucket_name)
    print(f'Bucket {bucket_name} Exists')
except ClientError as e:
    print(f'Creating bucket {bucket_name}')
    s3bucket = s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={ 'LocationConstraint': region_name }
    )

from utils.opensearch_roles import create_bedrock_execution_role, create_oss_policy_attach_bedrock_execution_role, create_policies_in_oss, interactive_sleep, create_os_client
vector_store_name = args.vectorstore_name
index_name = args.index_name
aoss_client = boto3_session.client('opensearchserverless')
bedrock_kb_execution_role = create_bedrock_execution_role(bucket_name=bucket_name)
bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
# create security, network and data access policies within OSS
encryption_policy, network_policy, access_policy = create_policies_in_oss(vector_store_name=vector_store_name,
                       aoss_client=aoss_client,
                       bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn)
collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')
pp.pprint(collection)


# Get the OpenSearch serverless collection URL
collection_id = collection['createCollectionDetail']['id']
host = collection_id + '.' + region_name + '.aoss.amazonaws.com'
print(host)

# wait for collection creation
# This can take couple of minutes to finish
response = aoss_client.batch_get_collection(names=[vector_store_name])
# Periodically check collection status
while (response['collectionDetails'][0]['status']) == 'CREATING':
    print('Creating collection...')
    interactive_sleep(30)
    response = aoss_client.batch_get_collection(names=[vector_store_name])
print('\nCollection successfully created:')
pp.pprint(response["collectionDetails"])

# create opensearch serverless access policy and attach it to Bedrock execution role
try:
    create_oss_policy_attach_bedrock_execution_role(collection_id=collection_id,
                                                    bedrock_kb_execution_role=bedrock_kb_execution_role)
    # It can take up to a minute for data access rules to be enforced
    interactive_sleep(60)
except Exception as e:
    print("Policy already exists")
    pp.pprint(e)


# Create the vector index in Opensearch serverless, with the knn_vector field index mapping, specifying the dimension size, name and engine.
oss_client = create_os_client(collection_id, region_name)

index_name = args.index_name
body_json = {
   "settings": {
      "index.knn": "true",
       "number_of_shards": 1,
       "knn.algo_param.ef_search": 512,
       "number_of_replicas": 0,
   },
   "mappings": {
      "properties": {
         "vector": {
            "type": "knn_vector",
            "dimension": 1536,
             "method": {
                 "name": "hnsw",
                 "engine": "faiss",
                 "space_type": "l2"
             },
         },
         "text": {
            "type": "text"
         },
         "text-metadata": {
            "type": "text"         
        }
      }
   }
}


# Create index
try:
    response = oss_client.indices.create(index=index_name, body=json.dumps(body_json))
    print('\nCreating index:')
    pp.pprint(response)

    # index creation can take up to a minute
    interactive_sleep(60)
except RequestError as e:
    # you can delete the index if its already exists
    # oss_client.indices.delete(index=index_name)
    print(f'Error while trying to create the index, with error {e.error}\nyou may unmark the delete above to delete, and recreate the index')

# Upload data to s3 to the bucket that was configured as a data source to the knowledge base
s3_client = boto3.client("s3")
def uploadDirectory(path,bucket_name):
        for root,dirs,files in os.walk(path):
            for file in files:
                s3_client.upload_file(os.path.join(root,file),bucket_name,file)

data_root = "../data/"
if args.copy_from_local:
    uploadDirectory(data_root, bucket_name)

opensearchServerlessConfiguration = {
            "collectionArn": collection["createCollectionDetail"]['arn'],
            "vectorIndexName": index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

# Ingest strategy - How to ingest data from the data source
chunkingStrategyConfiguration = {
    "chunkingStrategy": "FIXED_SIZE",
    "fixedSizeChunkingConfiguration": {
        "maxTokens": 512,
        "overlapPercentage": 20
    }
}

# The data source to ingest documents from, into the OpenSearch serverless knowledge base index
s3Configuration = {
    "bucketArn": f"arn:aws:s3:::{bucket_name}",
    # "inclusionPrefixes":["*.*"] # you can use this if you want to create a KB using data within s3 prefixes.
}

# The embedding model used by Bedrock to embed ingested documents, and realtime prompts
embeddingModelArn = f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"

name = args.knowledge_base_name
description = "Amazon shareholder letter knowledge base."
roleArn = bedrock_kb_execution_role_arn

# Create a KnowledgeBase
@retry(wait_random_min=1000, wait_random_max=2000,stop_max_attempt_number=7)
def create_knowledge_base_func():
    create_kb_response = bedrock_agent_client.create_knowledge_base(
        name = name,
        description = description,
        roleArn = roleArn,
        knowledgeBaseConfiguration = {
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embeddingModelArn
            }
        },
        storageConfiguration = {
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration":opensearchServerlessConfiguration
        }
    )
    return create_kb_response["knowledgeBase"]

try:
    kb = create_knowledge_base_func()
except Exception as err:
    print(f"{err=}, {type(err)=}")
pp.pprint(kb)

# Get KnowledgeBase 
get_kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId = kb['knowledgeBaseId'])

# Create a DataSource in KnowledgeBase 
create_ds_response = bedrock_agent_client.create_data_source(
    name = name,
    description = description,
    knowledgeBaseId = kb['knowledgeBaseId'],
    dataSourceConfiguration = {
        "type": "S3",
        "s3Configuration":s3Configuration
    },
    vectorIngestionConfiguration = {
        "chunkingConfiguration": chunkingStrategyConfiguration
    }
)
ds = create_ds_response["dataSource"]
pp.pprint(ds)
bedrock_agent_client.get_data_source(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])

# Start an ingestion job
start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])
job = start_job_response["ingestionJob"]
pp.pprint(job)

# Get job 
while(job['status']!='COMPLETE' ):
    get_job_response = bedrock_agent_client.get_ingestion_job(
      knowledgeBaseId = kb['knowledgeBaseId'],
        dataSourceId = ds["dataSourceId"],
        ingestionJobId = job["ingestionJobId"]
  )
    job = get_job_response["ingestionJob"]
pp.pprint(job)
interactive_sleep(40)

# Print the knowledge base Id in bedrock, that corresponds to the Opensearch index in the collection we created before, we will use it for the invocation later
kb_id = kb["knowledgeBaseId"]
pp.pprint(kb_id)

# Test the retrieve API
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name)

# retrieve api for fetching only the relevant context.
query = "What is Amazon's doing in the field of generative AI?"
relevant_documents = bedrock_agent_runtime_client.retrieve(
    retrievalQuery= {
        'text': query
    },
    knowledgeBaseId=kb_id,
    retrievalConfiguration= {
        'vectorSearchConfiguration': {
            'numberOfResults': 3 # will fetch top 3 documents which matches closely with the query.
        }
    }
)
pp.pprint(relevant_documents["retrievalResults"])

# Save required entries in a JSON file in order to be able to delete them later if needed
kb_info = {
    "ds_Id": ds["dataSourceId"],
    "kb_id": kb["knowledgeBaseId"],
    "index_name":args.index_name,
    "collection_id": collection_id,
    "access_policy_name": access_policy['accessPolicyDetail']['name'],
    "network_policy_name": network_policy['securityPolicyDetail']['name'],
    "encryption_policy_name": encryption_policy['securityPolicyDetail']['name'],
    "bucket_name": bucket_name,
    "region_name": region_name
}

with open("kb_info.json", "w") as file:
    json.dump(kb_info, file, indent=4)  # indent=4 for pretty-printing
