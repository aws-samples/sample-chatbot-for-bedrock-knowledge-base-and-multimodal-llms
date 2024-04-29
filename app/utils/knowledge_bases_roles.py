"""
This class is copied from https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/02_KnowledgeBases_and_RAG/utility.py
"""

import json
import os
import random
import time
import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection


def interactive_sleep(seconds: int):
    dots = ""
    for i in range(seconds):
        dots += "."
        print(dots, end="\r")
        time.sleep(1)
    print("Done!")


class KnowledgeBaseRoles:
    def __init__(self, region_name: str) -> None:
        self.suffix = random.randrange(200, 900)
        self.region_name = region_name
        self.boto3_session = boto3.session.Session(region_name=self.region_name)
        self.iam_client = self.boto3_session.client("iam")
        self.account_number = boto3.client("sts").get_caller_identity().get("Account")
        self.identity = boto3.client("sts").get_caller_identity()["Arn"]

        self.bedrock_execution_role_name = f"AmazonBedrockExecutionRoleForKnowledgeBase_{self.suffix}"
        self.fm_policy_name = f"AmazonBedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}"
        self.s3_policy_name = f"AmazonBedrockS3PolicyForKnowledgeBase_{self.suffix}"
        self.oss_policy_name = f"AmazonBedrockOSSPolicyForKnowledgeBase_{self.suffix}"


    def create_bedrock_execution_role(self, bucket_name: str) -> dict[str, dict[str, str]]:
        """
        Create an IAM role with necessary policies for Amazon Bedrock Knowledge Base Execution.

        Args:
            bucket_name (str): The name of the S3 bucket containing the documents.

        Returns:
            dict[str, dict[str, str]]: A dictionary containing the created IAM role details.
        """
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                    ],
                    "Resource": [
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/amazon.titan-embed-text-v1"
                    ],
                }
            ],
        }

        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                    "Condition": {
                        "StringEquals": {"aws:ResourceAccount": f"{self.account_number}"}
                    },
                }
            ],
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        # create policies based on the policy documents
        fm_policy = self.iam_client.create_policy(
            PolicyName=self.fm_policy_name,
            PolicyDocument=json.dumps(foundation_model_policy_document),
            Description="Policy for accessing foundation model",
        )

        s3_policy = self.iam_client.create_policy(
            PolicyName=self.s3_policy_name,
            PolicyDocument=json.dumps(s3_policy_document),
            Description="Policy for reading documents from s3",
        )

        # create bedrock execution role
        bedrock_kb_execution_role = self.iam_client.create_role(
            RoleName=self.bedrock_execution_role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3",
            MaxSessionDuration=3600,
        )

        # fetch arn of the policies and role created above
        bedrock_kb_execution_role_arn = bedrock_kb_execution_role["Role"]["Arn"]
        s3_policy_arn = s3_policy["Policy"]["Arn"]
        fm_policy_arn = fm_policy["Policy"]["Arn"]

        # attach policies to Amazon Bedrock execution role
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=fm_policy_arn
        )
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=s3_policy_arn
        )
        return bedrock_kb_execution_role


    def create_oss_policy_attach_bedrock_execution_role(
        self, collection_id: str, bedrock_kb_execution_role: dict[str, dict[str, str]]
    ) -> None:
        """
        Create an IAM policy for accessing OpenSearch Serverless and attach it to the Bedrock execution role.

        Args:
            collection_id (str): The ID of the OpenSearch Serverless collection.
            bedrock_kb_execution_role (dict[str, dict[str, str]]): The Bedrock execution role to attach the policy to.
        """
        # define oss policy document
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [
                        f"arn:aws:aoss:{self.region_name}:{self.account_number}:collection/{collection_id}"
                    ],
                }
            ],
        }
        oss_policy = self.iam_client.create_policy(
            PolicyName=self.oss_policy_name,
            PolicyDocument=json.dumps(oss_policy_document),
            Description="Policy for accessing opensearch serverless",
        )
        oss_policy_arn = oss_policy["Policy"]["Arn"]
        print("Opensearch serverless arn: ", oss_policy_arn)

        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=oss_policy_arn
        )


    def create_policies_in_oss(
        self,
        vector_store_name: str,
        aoss_client: boto3.client,
        bedrock_kb_execution_role_arn: str,
    ) -> tuple[dict, dict, dict]:
        """
        Create security, network, and data access policies in OpenSearch Serverless.

        Args:
            vector_store_name (str): The name of the vector store.
            aoss_client (boto3.client): The boto3 client for OpenSearch Serverless.
            bedrock_kb_execution_role_arn (str): The ARN of the Bedrock execution role.

        Returns:
            tuple[dict, dict, dict]: A tuple containing the encryption policy, network policy, and access policy dictionaries.
        """
        encryption_policy_name = f"bedrock-sample-rag-sp-{self.suffix}"
        network_policy_name = f"bedrock-sample-rag-np-{self.suffix}"
        access_policy_name = f"bedrock-sample-rag-ap-{self.suffix}"

        encryption_policy = aoss_client.create_security_policy(
            name=encryption_policy_name,
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "Resource": ["collection/" + vector_store_name],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
            type="encryption",
        )

        network_policy = aoss_client.create_security_policy(
            name=network_policy_name,
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "collection",
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
            type="network",
        )
        access_policy = aoss_client.create_access_policy(
            name=access_policy_name,
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/" + vector_store_name + "/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [self.identity, bedrock_kb_execution_role_arn],
                        "Description": "Easy data policy",
                    }
                ]
            ),
            type="data",
        )
        return encryption_policy, network_policy, access_policy



    def delete_iam_role_and_policies(self) -> None:
        fm_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.fm_policy_name}"
        s3_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.s3_policy_name}"
        oss_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.oss_policy_name}"
        self.iam_client.detach_role_policy(
            RoleName=self.bedrock_execution_role_name, PolicyArn=s3_policy_arn
        )
        self.iam_client.detach_role_policy(
            RoleName=self.bedrock_execution_role_name, PolicyArn=fm_policy_arn
        )
        self.iam_client.detach_role_policy(
            RoleName=self.bedrock_execution_role_name, PolicyArn=oss_policy_arn
        )
        self.iam_client.delete_role(RoleName=self.bedrock_execution_role_name)
        self.iam_client.delete_policy(PolicyArn=s3_policy_arn)
        self.iam_client.delete_policy(PolicyArn=fm_policy_arn)
        self.iam_client.delete_policy(PolicyArn=oss_policy_arn)
        return 0


    def create_os_client(self, collection_id: str):
        host = collection_id + "." + self.region_name + ".aoss.amazonaws.com"
        credentials = boto3.Session().get_credentials()
        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=AWSV4SignerAuth(credentials, self.region_name, "aoss"),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
        )
