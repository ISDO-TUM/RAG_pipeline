import csv
import datetime
import logging
import os
import tempfile
from typing import List, Tuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import pandas as pd
import seaborn as sns
from datasets import Dataset
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from matplotlib.colors import LinearSegmentedColormap
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rag.pipeline import Pipeline


class Evaluation:

    def __init__(self, pipeline: Pipeline, debug=False) -> None:
        self.pipeline = pipeline
        self.debug = debug
        self.delimiter = ';'

    def generate_question_answer_pairs(self, docs: Sequence[Document]) -> Dataset:
        """ 
        Generate question-answer pairs for evaluation using the sequence of documents.

        Parameters:
        docs (Sequence[Document]): A sequence of Document objects to use for generating question-answer pairs.

        Returns:
        Dataset: A Dataset object containing the generated question-answer pairs.
        """
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        distributions = {
            simple: 0.3,
            multi_context: 0.4,
            reasoning: 0.3
        }

        testset = generator.generate_with_langchain_docs(
            docs, 10, distributions)
        return testset.to_dataset()

    def get_evaluation_data_file(self, input_file: Optional[str]) -> Tuple[List[str], List[str]]:
        """
        Extracts questions and ground_truth from a specified file.
        If no file is provided uses default dummy values. 

        Parameters:
        eval_dataset_path (str): Path to the evaluation dataset file.

        Returns:
        Tuple[List[str], List[str]]: Lists of questions and ground truth answers.
        """
        questions = []
        ground_truth = []

        # default questions if file path not provided
        if not input_file:
            logging.info('Using default values')
            questions = ["Where are the 2024 Olympics held?"]

            ground_truth = ["The 2024 Olympics are held in Paris, France"]
        else:
            if not os.path.exists(input_file):
                raise FileNotFoundError(
                    f"The file {input_file} does not exist.")

            logging.info('Using CSV values')
            with open(input_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=';')

                for row in reader:
                    questions.append(row['question'])
                    ground_truth.append(row['ground_truth'])

        return questions, ground_truth

    def get_evaluation_data_file_streamlit(self, file_upload):
        file_content = file_upload.read().decode('utf-8').splitlines()
        questions = []
        ground_truth = []

        # Use csv.DictReader to read the CSV content
        reader = csv.DictReader(file_content, delimiter=';')

        # Extract data from each row
        for row in reader:
            questions.append(row['question'])
            ground_truth.append(row['ground_truth'])

        return questions, ground_truth

    def generate_response(self, test_questions: list, test_answers: list, ) -> Dataset:
        """
        Generate responses using the specified model and prompt.

        Parameters:
        test_questions List[str]: List of questions to be used for generating responses.
        test_answers [List[str]]: List of ground truth answers for evaluation.
        model (str): The model to use for generating responses.
        prompt (str): The prompt to use for generating responses.

        Returns:
        Dataset: A Dataset object containing the questions, contexts, answers, and ground truth answers.
        """

        answers = []
        contexts = []

        for question in test_questions:
            state = self.pipeline.retrieve(query=question, conversation=[])

            answer = state["result"]
            context = state["retrieved_documents"]

            answers.append(answer)
            contexts.append(context)

        contexts = [[doc.page_content for doc in sublist]
                    for sublist in contexts]

        dataset_dict = {"question": test_questions,
                        "contexts": contexts, "answer": answers}

        if test_answers is not None:
            dataset_dict["ground_truth"] = test_answers

        ds = Dataset.from_dict(dataset_dict)

        if self.debug:
            print(contexts)
            print(ds)

        print("[INFO] Created Dataset for evaluation.")

        return ds

    def perform_evaluation(self, eval_dataset_path: Union[str, None], file_upload=False) -> dict:
        """
        Perform evaluation on the given dataset path.

        Parameters:
        eval_dataset_path (str): Path to the evaluation dataset file.

        Returns:
        dict: A dictionary containing the evaluation results.
        """
        if file_upload:
            questions, ground_truth = self.get_evaluation_data_file_streamlit(file_upload)
        else:
            questions, ground_truth = self.get_evaluation_data_file(eval_dataset_path)

        ds = self.generate_response(
            questions, ground_truth
        )

        metrics = [context_recall, context_precision,
                   faithfulness, answer_relevancy]
        results = evaluate(
            ds,
            metrics=metrics,
            raise_exceptions=False
        )
        print("[INFO] Results are:\n", pd.DataFrame(results, index=[0]))
        return results, ds

    def to_csv(self, dataset: Dataset) -> None:
        """
        Save the retrieved dataset to a CSV file.

        Parameters:
        ds (Dataset): The dataset to save.
        """
        df = dataset.to_pandas()
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, "df_evaluation.csv")
        df.to_csv(file_path, sep=self.delimiter)
        if os.path.exists(file_path):
            print("[INFO] Created CSV file successfully.")
        else:
            print("[Error]: CSV file was not saved.")
        return

    def log_on_mlflow(self, results: dict) -> None:
        """
        Log evaluation results and configuration parameters to MLflow.

        Parameters:
        results (Dict[str, float]): A dictionary containing the evaluation results with metric names as keys and their values as values.
        """
        print("[INFO] Starting ML Logging.")
        mlflow.set_experiment("Capstone - RAG")
        with mlflow.start_run():  # mlflow ui --port 5000
            mlflow.set_tag("mlflow.user", "capstone_worker")
            mlflow.set_tag("mlflow.runName", "Test 00")

            mlflow.log_artifact("./config.ini")

            # log config
            for section in self.pipeline.config.sections():
                for key in self.pipeline.config[section]:
                    param_name = f"{section}.{key}"
                    param_value = self.pipeline.config[section][key]
                    mlflow.log_param(param_name, param_value)

            for metric_name, metric_value in results.items():
                mlflow.log_metric(metric_name, metric_value)

        print("[INFO] ML Logging finished.")

    def generate_heatmap(self, results: dict) -> str:
        """
        Generate a heatmap from the evaluation results.

        Parameters:
        results (Dict[str, float]): A dictionary containing the evaluation results.

        Returns:
        str: The file path to the saved heatmap image.
        """
        df = results.to_pandas()
        heatmap = df[['answer_relevancy', 'faithfulness',
                      'context_precision', 'context_recall']]
        cmap = LinearSegmentedColormap.from_list('green_red', ["red", "green"])

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
        plt.yticks(ticks=range(len(df['question'])),
                   labels=df['question'], rotation=0)

        # Save the heatmap plot as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight')
            image_filename = tmpfile.name
        plt.close()
        return image_filename

    def generate_table(self, results: dict) -> Table:
        """
        Generate a pdf table from the evaluation results.

        Parameters:
        results (Dict[str, Any]): A dictionary containing the evaluation results.

        Returns:
        Table: A ReportLab Table object representing the evaluation results.
        """
        keys = list(results.keys())
        # Convert values to strings
        values = [str(value) for value in results.values()]
        table_data = [['Metric', 'Value']] + [[key, value]
                                              for key, value in zip(keys, values)]

        # Create a table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        return table

    def generate_report(self, results: str, ds: Dataset) -> None:
        """
        Generates a PDF report from the evaluation results.
        Parameters:
        results (Dict[str, Any]): A dictionary containing the evaluation results.
        ds (Dataset): A Dataset object containing the questions, contexts, answers and ground truth answers.
        """

        # add the results to a csv file.
        self.to_csv(ds)

        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = os.path.join(output_dir, f"report_{current_time}.pdf")
        title = "Evaluation Results"

        doc = SimpleDocTemplate(report_file, pagesize=letter)
        elements = []

        # Set the title
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        normal_style = styles['Normal']

        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 5))  # Adjust the space as needed
        elements.append(HRFlowable(width="100%", thickness=1,
                                   lineCap='round', color=colors.black))

        # Add configuration details
        elements.append(Paragraph("Configuration:", styles['Heading3']))
        for section_name, section in self.pipeline.config.items():
            elements.append(Paragraph(f"{section_name}:", normal_style))
            for key, value in section.items():
                elements.append(Paragraph(f"    {key}: {value}", normal_style))

        # render the table for metrics
        title = Paragraph("<b>Results</b>", styles['Heading3'])
        elements.append(title)
        table = self.generate_table(results)
        elements.append(table)
        elements.append(Spacer(1, 10))

        # Create the heatmap
        subtitle = Paragraph("<b>HeatMap per Question</b>", styles['Heading3'])
        elements.append(subtitle)
        elements.append(Spacer(1, 5))
        image_filename = self.generate_heatmap(results)
        elements.append(Image(image_filename, width=400, height=200))

        # Build the PDF
        doc.build(elements)
        print(f"Report generated successfully and saved as {report_file}")
