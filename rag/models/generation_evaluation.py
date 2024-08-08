import copy
import json
import os
import sys
from datetime import datetime
from typing import List

from rag.fixtures.prompts import system_prompt_templates, user_prompt_templates
from rag.models.chatbot import Chatbot

"""
This script evaluates the generation. It tests both LLMs and Prompt Templates for various metrics:
---------------
Context Fidelity:
This metric captures how faithful the LLM is to the given context. It is tested by purposefully supplying the LLM with false information and testing
if it recites this information
Test Cases:
prompt: example question that could come from a user
document: document with purposefully WRONG information (context)
answer: answer based on the WRONG information from the document
-> The LLM should answer the similar to "answer" and stay true to the context
---------------
"""


class Document:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.metadata = {"source": source}

    def __str__(self) -> str:
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"


def extract_test_case_documents(test_case) -> List[Document]:
    documents = []
    for doc in test_case["documents"]:
        page_content = doc["page_content"]
        source = doc["source"]
        document = Document(page_content, source)
        documents.append(document)
    return documents


def build_documents_string(documents: List[Document]) -> str:
    """
    Builds a string representation of the given list of documents.

    Params:
        documents: the documents
    Returns:
        the single string representation
    """
    document_string = ""
    for doc in documents:
        document_string += str(doc) + "\n"
    return document_string


def get_llm_report_for_test(chatbot: Chatbot, test_case, system_prompt: str, test_name: str) -> dict:
    """
    Builds a report with the chatbot response based on a test

    Params:
        chatbot: the chatbot to use
        test_case: the test case to evaluate
        system_prompt: the system prompt
        test_name: the test name
    Returns:
        report containing chatbot response
    """
    question = test_case["question"]

    documents = extract_test_case_documents(test_case)
    document_string = build_documents_string(documents)

    if test_name in user_prompt_templates["testing"]:
        user_prompt = (user_prompt_templates["testing"][test_name]
                       .substitute(question=question, documents=document_string))
    else:
        user_prompt = question

    answer = chatbot.custom_prompt(system_prompt, user_prompt)

    return {
        "test_case": test_case,
        "system_prompt_template": system_prompt,
        "user_prompt_template": user_prompt,
        "documents": documents,
        "llm_response": answer,
        "evaluated": False,
        "passed": False,
    }


def store_results(output_folder: str, test_result):
    """
    Stores the test results in a given output folder with the file name test_{test_name}_-{date}

    Params:
        output_folder: the folder to store the results in
        test_result: the result
    """
    output_path = os.path.join(output_folder,
                               f'test_{test_result["test_name"]}_-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json')
    # Create folder if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as file:
        json.dump(test_result, file, indent=4)


def get_user_input(question) -> bool:
    """
    Asks the user a yes or no questions and returns his answer.
    The user can also cancel, which terminates the program

    Params:
        question: the question to ask the user
    Returns:
        whether the user said yes or no
    """
    user_input = ""
    yes = ["yes", "y"]
    no = ["no", "n"]
    exit_program = ["exit", "e"]

    while user_input.lower() not in yes + no + exit_program:
        user_input = input(question)

    if user_input.lower() in yes:
        return True
    elif user_input.lower() in no:
        return False
    else:
        sys.exit(0)


class GenerationTester:
    """
    Tester for the generation part of the pipeline
    """

    def __init__(self, chatbots: List[Chatbot], output_folder="gen_test_output"):
        self.output_folder = output_folder
        self.chatbots = chatbots

    def test_and_score(self, test_cases_path: str):
        """
        Runs the tests of a given path, evaluates and scores them.

        Params:
            test_cases_path: path containing the test cases
        """
        with open(test_cases_path, 'r') as file:
            test_cases = json.load(file)

            test_results = self.test_llm_answers(test_cases)
        evaluation_results = self.evaluate_tests(test_results)

        self.calculate_and_print_scores(evaluation_results)

        store_results(self.output_folder, evaluation_results)
        print(f"""------------------------------------
Test evaluation result are saved in: {self.output_folder}""")

    def test_llm_answers(self, test_cases: dict) -> (str, str):
        """
        This method creates a json for all test cases and the answer of the chatbots when prompted

        Args:
            test_cases: the test cases json
        Returns:
            the generated json and the path of the generated json
        """

        test_name = test_cases["test_name"]
        test_result = {
            "test_name": test_name,
            "test_description": test_cases["test_description"],
            "results": {}
        }

        for chatbot in self.chatbots:
            test_result["results"][chatbot.model_name] = {}
            for index, system_prompt in enumerate(system_prompt_templates["testing"][test_name]):
                test_result["results"][chatbot.model_name][index] = []
                for test_case in test_cases["test_cases"]:
                    report = get_llm_report_for_test(chatbot, test_case, system_prompt, test_name)

                    test_result["results"][chatbot.model_name][index].append(report)

        return test_result

    @staticmethod
    def evaluate_tests(test_results: dict) -> (str, str):
        """
        Using the json containing the llm answers generated by the test method, add evaluation whether the answer is
        correct or not. This is done manually, except for rag relevance

        Returns:
            the json containing the evaluation and the path to it
        """
        evaluation_results = copy.deepcopy(test_results)

        test_name = test_results["test_name"]
        test_description = test_results["test_description"]
        chatbots = test_results["results"]

        print(f"Test Name:\n{test_name}")
        print(f"Test Description:\n{test_description} ")
        print("To evaluate: y=passed, n=not passed, e=end evaluation\n")

        for chatbot in chatbots:
            print(f"Chatbot: {chatbot}")

            for system_prompt_index in chatbots[chatbot]:
                system_prompt = system_prompt_templates["testing"][test_name]
                print(f"System Prompt #{system_prompt_index}: {system_prompt}")

                system_prompt_evaluated_results = []

                for test in chatbots[chatbot][system_prompt_index]:
                    question = test["test_case"]["question"]
                    documents = test["test_case"]["documents"]
                    correct_answer = test["test_case"]["answer"]
                    llm_response = test["llm_response"]

                    print(f"""------------------------------------
Question: {question}""")

                    if documents:
                        print(f"Documents: {', '.join(documents)}")

                    print(f"LLM Response: {llm_response}")

                    if correct_answer:
                        print(f"Correct Answer: {correct_answer}")

                    if test_name in ["detect_rag_irrelevance", "detect_rag_relevance", "guardrail_work",
                                     "guardrail_pii"]:
                        passed = correct_answer.lower() in llm_response.lower()
                    else:
                        passed = get_user_input("Did the LLM answer correctly? "
                                                "[y=passed, n=not passed, e=end evaluation] ")

                    test["evaluated"] = True
                    test["passed"] = passed
                    system_prompt_evaluated_results.append(test)

                evaluation_results["results"][chatbot][system_prompt_index] = system_prompt_evaluated_results

        return evaluation_results

    @staticmethod
    def calculate_and_print_scores(evaluation_results: dict):
        """
        Scores given evaluation results and prints out the scores.

        Params:
            evaluation_results: the evaluation results to score
        """
        test_name = evaluation_results["test_name"]
        test_description = evaluation_results["test_description"]
        chatbots = evaluation_results["results"]
        not_evaluated = 0

        print(f"""------------------------------------
Test Name: {test_name}
Test Description: {test_description}""")

        for chatbot in chatbots:
            for system_prompt_index in chatbots[chatbot]:
                total = len(chatbots[chatbot][system_prompt_index])
                passed = 0
                for test in chatbots[chatbot][system_prompt_index]:
                    if test["evaluated"] and test["passed"]:
                        passed += 1
                    elif not test["evaluated"]:
                        not_evaluated += 1

                print(f"""LLM: {chatbot} | Prompt #{system_prompt_index}
Score: {(passed / total) * 100}%""")

        if not_evaluated > 0:
            print(f"Test's that aren't evaluated: {not_evaluated}")
