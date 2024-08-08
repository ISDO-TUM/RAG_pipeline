import warnings
import dotenv
import configparser
from rag.models.evaluation import Evaluation
from rag.fixtures.prompts import initial_conversation
from rag.functions.vector_indexing import get_vectordb
from rag.pipeline import Pipeline
import argparse
import logging as logging
from rag.models.dataloader import DataLoader


def load_pipeline():
    config = configparser.ConfigParser()
    config.read("config.ini")
    dotenv.load_dotenv()
    warnings.simplefilter("ignore", category=FutureWarning)

    # Initial conversation
    conversation = initial_conversation

    # Set up Pipeline
    data_loader = DataLoader(config["ingestion"])
    vectordb = get_vectordb(config, data_loader)
    pipeline = Pipeline(vectordb=vectordb, config=config)
    print("[INFO] Pipeline created.")
    eval=Evaluation(pipeline=pipeline, debug=False)
    
    return eval
    

def main():
    eval=load_pipeline()
    parser = argparse.ArgumentParser(description="Tool for evaluation")
    
    # Add arguments
    parser.add_argument("-i", "--csv", type=str, help="Path of the input csv file")
    parser.add_argument("-o", "--output", action="store_true", help="generate output file")
    parser.add_argument(
        "-m", "--mlflow", action="store_true", help="whether to log on mlflow or not"
    )
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.csv:
        results, ds = eval.perform_evaluation(eval_dataset_path=args.csv)
    else:
        results, ds = eval.perform_evaluation(eval_dataset_path=None)
    
    print(results)
    if args.output:
        eval.generate_report(results, ds)

    if args.mlflow:
        eval.log_on_mlflow(results=results)


main()
