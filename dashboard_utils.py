import configparser
import numpy as np
from typing import Any, Dict
from rag.models.evaluation import Evaluation
import warnings
import os
import dotenv
import configparser
from rag.functions.vector_indexing import get_vectordb
from rag.pipeline import Pipeline
import requests
from pymongo import MongoClient
from ragas.evaluation import Result
from rag.models.dataloader import DataLoader


# ----------------- Initializations -----------------
# load environment variables
dotenv.load_dotenv()
firebase_api_key = os.environ.get("FIREBASE_API_KEY")
atlas_uri = os.environ.get("ATLAS_URI")
db_name = os.environ.get("DB_NAME")
collection_name = os.environ.get("COLLECTION_NAME")


# ----------------- Authentication -----------------
def authenticate_user(email: str, password: str) -> bool:
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


# ----------------- Configuration -----------------
def update_config(values):
    textsplitter = values[1]
    third_indexing_param = "textsplitter_semantic_breakpoint_type" if textsplitter == "SemanticTextSplitter" else "textsplitter_recursive_chunk_size"
    config_dic = {
        "indexing": {
            "embeddings": values[0],
            "textsplitter": values[1],
            third_indexing_param: values[2],
        },
        "retrieval": {
            "method": values[3],
            "k_chunks": values[4],
            "reranker": values[5]
        },
        "chatbot": {
            "model": values[6],
            "prompt": values[7],
            "temperature": values[8],
        },
    }
    
    # Load the config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Update the config values
    for section, parameters in config_dic.items():
        if section in config:
            for key, value in parameters.items():
                config[section][key] = str(value)
    # Save the config file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    return config_dic


# ----------------- Pipeline -----------------
def call_pipeline(file_upload):
    config = configparser.ConfigParser()
    config.read("config.ini")
    warnings.simplefilter("ignore", category=FutureWarning)

    # Set up Pipeline
    data_loader = DataLoader(config["ingestion"])
    vectordb = get_vectordb(config, data_loader)
    pipeline = Pipeline(vectordb=vectordb, config=config)
    print("[INFO] Pipeline created.")
    eval = Evaluation(pipeline=pipeline, debug=False)
    results, _ = eval.perform_evaluation(
        eval_dataset_path=None, file_upload=file_upload
    )
    return results


# ----------------- MongoDB -----------------
def startup_mongodb_client():
    mongodb_client = MongoClient(atlas_uri)
    col = mongodb_client[db_name][collection_name]
    print("[INFO] MongoDB client started.")
    return mongodb_client, col

def close_mongodb_client(mongodb_client):
    mongodb_client.close()
    print("[INFO] MongoDB client closed.")
    
def insert_document(test_name: str, timestamp: Any, config: Dict[str, Any], results: Result) -> bool:
    average_score_per_metric = results
    average_score = sum(average_score_per_metric.values()) / len(average_score_per_metric)
    results_df = results.to_pandas()
    
    results_per_question = []
    for index, row in results_df.iterrows():
        row_dict = row.to_dict()
        
        # Convert numpy arrays to lists, in order to encode them to JSON
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                row_dict[key] = value.tolist()
        results_per_question.append(row_dict)
    
    try: 
        mongodb_client, col = startup_mongodb_client()
        post = {
            "test_name": test_name,
            "timestamp": timestamp,
            "config": config,
            "results": {
                "average_score": average_score,
                "average_score_per_metric": average_score_per_metric,
                "results_per_question": results_per_question
            }
        }
        col.insert_one(post)
        close_mongodb_client(mongodb_client)
    except Exception as e:
        print(e)
        return False
    
    return True

def get_data(search_query=None):
    try:
        mongodb_client, col = startup_mongodb_client()
        if search_query:
            query = {"test_name": {"$regex": search_query, "$options": "i"}}
        else:
            query = {}

        data = list(col.find(query).sort("results.average_score", -1))
        close_mongodb_client(mongodb_client)
    except Exception as e:
        print(e)
        return False
    
    return data
