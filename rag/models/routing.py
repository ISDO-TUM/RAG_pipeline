from typing import List

from langchain_core.documents import Document

from rag.fixtures.prompts import system_prompt_templates, user_prompt_templates
from rag.models.chatbot import get_chatbot
from rag.functions.logger import CustomLogger

class Routing:
    """
    Class containing routers for  the pipeline
    """

    def __init__(self, config):
        self.routing_prams = config["routing"]
        self.chatbot = get_chatbot(config, config["routing"]["provider"], config["routing"]["model"], None)
        self.logger = CustomLogger("[ROUTING]", config["logging"]["filename"])
        self.logger.log("Routing initialised")

    def rag_relevance(self, query: str) -> str:
        """
        Router that detects whether a question needs RAG for an answer.

        Params:
            query: the query
        Returns:
            "rag" or "no_rag" based on whether rag is needed
        """
        if "COMPANY_NAME" in query.lower():
            self.logger.log("Retrieval is needed based on keyword analysis")
            return "rag"

        answer = self.chatbot.custom_prompt(system_prompt_templates["routing"]["rag_relevance"], query)

        if "true" in answer.lower():
            self.logger.log("Retrieval is needed based on Routing LLM")
            return "rag"
        else:
            self.logger.log("Retrieval is Skipped based on Routing LLM")
            return "no_rag"

    def document_relevance(self, query: str, documents: List[Document]) -> bool:
        """
        Router that detects whether the documents are relevant to the questions.

        Params:
            query: the query
            documents: the documents
        Returns:
            whether the document are relevant to the questions
        """
        answer = self.chatbot.custom_prompt(system_prompt_templates["routing"]["document_relevance"],
                                            user_prompt_templates["routing"]["document_relevance"]
                                            .substitute(question=query, documents=documents))

        if "true" in answer.lower():
            self.logger.log("The LLM has decided that the documents help to answer the question")
            return True
        else:
            self.logger.log("The LLM has decided that the documents DO NOT help to answer the question")
            return False

    def hallucination_detection(self, query: str, documents: List[Document], llm_answer: str) -> str:
        """
        Router that detects whether the answer to the question is based on the documents.

        Params:
            query: the query
            documents: the documents
            llm_answer: the answer of the chatbot
        Returns:
            "hallucination" or "factual"
        """
        answer = self.chatbot.custom_prompt(system_prompt_templates["routing"]["hallucination_detection"],
                                            user_prompt_templates["routing"]["hallucination_detection"]
                                            .substitute(question=query, documents=documents, llm_answer=llm_answer))

        if "true" in answer.lower():
            self.logger.log("The LLM has decided that the generation is factual and not hallucinated")
            return "factual"
        else:
            self.logger.log("The LLM has decided that the generation is NOT factual and potentially hallucinated")
            return "hallucination"

    def guardrail_work(self, query: str) -> str:
        """
        Simple router checking if a query is work related

        Params:
            query: the query
        Returns:
            "work_related" or "not_work_related"
        """
        answer = self.chatbot.custom_prompt(system_prompt_templates["routing"]["guardrail_work"], query)

        if "true" in answer.lower():
            self.logger.log("The LLM guardrail has decided that the question is work related")
            return "work_related"
        else:
            self.logger.log("The LLM guardrail has decided that the question is not work related")
            return "not_work_related"

    def guardrail_pii(self, query: str) -> str:
        """
        Simple router checking if a query contains personal identifying information.

        Params:
            query: the query
        Returns:
            "contains_pii" or "not_contains_pii"
        """
        answer = self.chatbot.custom_prompt(system_prompt_templates["routing"]["guardrail_pii"],
                                            query)

        if "true" in answer.lower():
            self.logger.log("The LLM guardrail has decided that the question does contain pii")
            return "contains_pii"
        else:
            self.logger.log("The LLM guardrail has decided that the question does not contain pii")
            return "not_contains_pii"

    def translate_query(self, query: str) -> str:
        """
        Router that translates the query to make the question better to understand for an llm

        Params:
            query: the query
        Returns:
             the translated query
        """
        answer = self.chatbot.custom_prompt("", system_prompt_templates["routing"]["query_translation"] + query)

        self.logger.log("Query has been refined")
        # TODO: delete this block after testing
        # Start of block
        self.logger.log("[OLD QUERY] " + query)
        self.logger.log("[TRANSLATED QUERY] " + answer)
        # End of block
        return answer
