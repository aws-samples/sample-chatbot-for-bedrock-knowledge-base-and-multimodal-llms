"""
This file is copied from: https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/02_KnowledgeBases_and_RAG/4_CLEAN_UP.ipynb
"""

import json
import boto3
import sys

sys.path.append("..")  # Add the parent directory to the Python path
from utils.knowledge_bases_roles import KnowledgeBaseRoles, KBInfo


with open("kb_info.json") as f:
    kb_info = KBInfo.parse_obj(json.load(f))

boto3_session = boto3.session.Session(region_name=kb_info.region_name)
bedrock_agent_client = boto3_session.client(
    "bedrock-agent", region_name=kb_info.region_name
)
s3_client = boto3.client("s3", region_name=kb_info.region_name)

kb_roles = KnowledgeBaseRoles(kb_info.region_name)
oss_client = kb_roles.create_os_client(kb_info.collection_id)
aoss_client = boto3_session.client("opensearchserverless")

bedrock_agent_client.delete_data_source(
    dataSourceId=kb_info.ds_id, knowledgeBaseId=kb_info.kb_id
)
bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=kb_info.kb_id)
oss_client.indices.delete(index=kb_info.index_name)
aoss_client.delete_collection(id=kb_info.collection_id)
aoss_client.delete_access_policy(type="data", name=kb_info.access_policy_name)
aoss_client.delete_security_policy(type="network", name=kb_info.network_policy_name)
aoss_client.delete_security_policy(
    type="encryption", name=kb_info.encryption_policy_name
)

bucket_name = kb_info.bucket_name
objects = s3_client.list_objects(Bucket=bucket_name)
if "Contents" in objects:
    for obj in objects["Contents"]:
        s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
s3_client.delete_bucket(Bucket=bucket_name)
