"""
This class is copied from https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/02_KnowledgeBases_and_RAG/4_CLEAN_UP.ipynb
"""

import argparse
import json
import boto3
from knowledge_bases_roles import KnowledgeBaseRoles, KBInfo
from pathlib import Path


def delete_bucket(bucket_name: str, s3_client: boto3.client) -> None:
    """
    Delete an S3 bucket and all objects within it.
    Args:
      bucket_name: The name of the bucket to be deleted.
    """
    objects = s3_client.list_objects(Bucket=bucket_name, MaxKeys=1000)
    if "Contents" in objects:
        while objects["Contents"]:
            keys = [obj["Key"] for obj in objects["Contents"]]
            s3_client.delete_objects(
                Bucket=bucket_name, Delete={"Objects": [{"Key": key} for key in keys]}
            )
            if "NextContinuationToken" in objects:
                objects = s3_client.list_objects(
                    Bucket=bucket_name,
                    MaxKeys=1000,
                    ContinuationToken=objects["NextContinuationToken"],
                )
            else:
                break

    s3_client.delete_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' has been deleted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deletes a knowledge base with a specified name"
    )
    parser.add_argument(
        "--knowledge_base_name", type=str, required=True, help="Knowledge base name"
    )
    args = parser.parse_args()
    path = Path(__file__).parent.absolute()  # gets path of parent directory
    with open(path / f"{args.knowledge_base_name}.json", encoding="utf-8") as f:
        kb_info = KBInfo.parse_obj(json.load(f))

    boto3_session = boto3.session.Session(region_name=kb_info.region_name)
    bedrock_agent_client = boto3_session.client(
        "bedrock-agent", region_name=kb_info.region_name
    )
    s3_client = boto3_session.client("s3")

    kb_roles = KnowledgeBaseRoles(
        kb_info.region_name,
        bedrock_execution_role_name=kb_info.bedrock_execution_role_name,
        fm_policy_name=kb_info.fm_policy_name,
        s3_policy_name=kb_info.s3_policy_name,
        oss_policy_name=kb_info.oss_policy_name,
    )
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

    kb_roles.delete_iam_role_and_policies()
    delete_bucket(kb_info.bucket_name, s3_client)
