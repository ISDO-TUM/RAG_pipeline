from typing import List

from langchain_core.documents import Document

from rag.models.chatbot import get_chatbot
from rag.fixtures.prompts import system_prompt_templates
from rag.functions.logger import CustomLogger
class Generation:
    """
    Class for the generation part of the pipeline
    """

    def __init__(self, config):
        self.chatbot = get_chatbot(
                                config = config, 
                                provider = config["generation"]["provider"], 
                                model = config["generation"]["model"],
                                default_system_prompt = str(system_prompt_templates["generation"]["no_rag"]))
        
        
        self.logger = CustomLogger("[GENERATION]", config["logging"]["filename"])
        self.logger.log("Generation initialised")
        


    def generate_rag(self, query, conversation, retrieved_documents):
        """
        Generate a chatbot response

        Params:
            query: the query
            conversation: the previous conversation
            retrieved_documents: the documents accompanying the query
        Returns:
            the chatbot response
        """
        if retrieved_documents:
            prompt_template = system_prompt_templates["generation"]["rag_with_docs"]
            documents_string = self.documents_to_string(retrieved_documents)
            system_prompt = prompt_template.substitute(documents=documents_string)

            result = self.chatbot.custom_prompt(system_prompt, query)
            self.logger.log(f"[INFO] LLM Response: {result}")
            return result
        
        else:
            system_prompt = system_prompt_templates["generation"]["rag_no_docs"].substitute()
            
            result = self.chatbot.custom_prompt(system_prompt, query)
            self.logger.log(f"[LLM RESPONSE] {result}")
            return result
        


    def generate_no_rag(self, query, conversation):
        """
        Generate a chatbot response

        Params:
            query: the query
            conversation: the previous conversation
            retrieved_documents: the documents accompanying the query
        Returns:
            the chatbot response
        """
        result = self.chatbot.answer_question(query)
        self.logger.log(f"[LLM RESPONSE] {result}")
        return result


    def documents_to_string(self, documents):
        documents_string = ""
        for d in documents:
            documents_string += f"{d.page_content}\nSource: {d.metadata['source']}"
        return documents_string
