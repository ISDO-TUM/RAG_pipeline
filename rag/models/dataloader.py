import os
import json
import boto3
from typing import List, Dict
from langchain_core.documents.base import Document
from azure.storage.blob import ContainerClient

def _metadata_func(record: Dict, metadata: Dict) -> Dict:
    """Extracts and adds metadata to odd to documents

    Args:
        record: the dict containing the content of the json.
        metadata: the dict containing the extracted metadata.
    Returns:
        dict: the updated metadata dict
    """
    metadata["source"] = record.get("url")
    metadata["date"] = record.get("lastRetrievalTime")
    metadata["title"] = record.get("title")
    metadata["type"] = record.get("type")
    metadata["language"] = record.get("language")
    return metadata

def _parse_json_by_language(documents: List[Document], json_data: Dict, language: str = "en"):
    """Converts a json to a langchain document if it's of given language 
       and appends it to the documents list

    Args:
        documents: the list of documents to append the converted document to
        json_data: the json data to convert
    """
    try:
        if json_data['language'] == language:
            documents.append(Document(page_content=json_data['content'], metadata=_metadata_func(json_data, {})))
    except KeyError:
        pass
class DataLoader:
    def __init__(self, config):
        self.config = config

    """
    Loads the data from the given source into langchain documents
    Returns:
        documents: the langchain documents to embed
    """
    def load_data(self) -> List[Document]:
        if self.config["method"] == "local":
            return self._load_from_local(self.config["data_folder"])
        elif self.config["method"] == "azure_blob_storage":
            return self._load_from_azure(self.config["azure_container_sas_url"])
        elif self.config["method"] == "aws_s3":
            return self._load_from_aws_s3(self.config["aws_region_name"],
                                          self.config["aws_bucket_name"],
                                          self.config["aws_access_key_id"],
                                          self.config["aws_secret_access_key"])
            
    def _load_from_local(self, data_folder: str) -> List[Document]:
        print(f"[INFO] Loading data from {data_folder}")
        documents = []
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        json_data = json.load(f)
                        _parse_json_by_language(documents, json_data, self.config["language"])
        print("[INFO] Data loaded.")
        return documents

    def _load_from_azure(self, container_sas_url) -> List[Document]:
        print("[INFO] Loading data from azure blob storage")
        documents = []
        container_client = ContainerClient.from_container_url(container_url=container_sas_url)
        blobs = container_client.list_blobs()
        for blob in blobs:
            if blob.name.endswith('.json'):
                blob_client = container_client.get_blob_client(blob.name)
                json_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))
                _parse_json_by_language(documents, json_data, self.config["language"])
        
        print("[INFO] Data loaded.")
        return documents

    def _load_from_aws_s3(self, region_name, bucket_name, aws_access_key_id, aws_secret_access_key) -> List[Document]:
        print("[INFO] Loading data from aws s3 bucket")
        documents = []
        s3_client = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    obj_data = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                    json_data = json.loads(obj_data['Body'].read().decode('utf-8'))
                    _parse_json_by_language(documents, json_data, self.config["language"])
        
        print("[INFO] Data loaded.")
        return documents