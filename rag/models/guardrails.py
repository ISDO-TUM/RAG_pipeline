from enum import Enum

from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (case_insensitive_matching_strategy, combined_exact_fuzzy_matching_strategy, fuzzy_matching_strategy)
from presidio_analyzer import Pattern, PatternRecognizer
from rag.fixtures.prompts import system_prompt_templates, guardrail_responses, user_prompt_templates
from rag.models.chatbot import get_chatbot
from rag.functions.logger import CustomLogger
import json

class GuardrailResponse(Enum):
    OK = "ok"
    NOT_OK = "not_ok"
    CHANGED = "changed"

class Guardrails:

    def __init__(self, config):
        self.guardrail_params = config["guardrails"]

        # init chatbot
        self.chatbot = get_chatbot(config, config["guardrails"]["provider"], config["guardrails"]["model"], None)

        # init logger
        self.logger = CustomLogger("[GUARDRAIL]", config["logging"]["filename"])
        self.logger.log("Guardrails initialised")

        # init "blockers"
        self.block_pii = config["guardrails"]["block_pii"].lower() == "true"
        self.block_not_work_related = config["guardrails"]["block_not_work_related"].lower() == "true"

        # init "anonymizers"
        self.anonymize_pii = (config["guardrails"]["anonymize_pii"].lower() == "true") and (not self.block_pii)

        if self.anonymize_pii:
            self.anonymizer = PresidioReversibleAnonymizer(analyzed_fields=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"])
            self.deanonymization_mapping = None
            self.exclude_from_anonymization = ["COMPANY_NAME"]


    def guardrail_input(self, query):
        """
        Apply input guardrails to the given query.

        This method processes the input query through various checks and transformations,
        including PII detection, work-relatedness verification, and anonymization if configured.

        Args:
            query (str): The original input query from the user.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - response_type (GuardrailResponse): Enum indicating the result of guardrail checks.
                - query (str): The potentially modified query after applying guardrails.
                - error_message (str, optional): Explanation if the query is rejected, None otherwise.

        The response_type can be:
            - GuardrailResponse.OK: Query passed all checks without modification.
            - GuardrailResponse.NOT_OK: Query failed checks and should not be processed.
            - GuardrailResponse.CHANGED: Query was modified (e.g., anonymized) and can be processed.
        """

        if self.block_pii:
            answer = self.chatbot.custom_prompt(system_prompt_templates["guardrails"]["guardrail_pii"], user_prompt_templates["guardrails"]["pii"].substitute(query=query))
            
            if "true" in answer.lower():
                self.logger.log("The LLM guardrail has decided that the question does contain pii")                    
                return GuardrailResponse.NOT_OK, query, guardrail_responses["contains_pii"]
            
            else:
                self.logger.log("The LLM guardrail has decided that the question does NOT contain pii")

        if self.block_not_work_related:
            answer = self.chatbot.custom_prompt(system_prompt_templates["guardrails"]["guardrail_work"], user_prompt_templates["guardrails"]["work_related"].substitute(query=query))

            if "true" in answer.lower():
                self.logger.log("The LLM guardrail has decided that the question is work related")
            else:
                self.logger.log("The LLM guardrail has decided that the question is NOT work related")
                return GuardrailResponse.NOT_OK, query, guardrail_responses["not_work_related"]

        if self.anonymize_pii:
            new_query = query

            if self.guardrail_params["anonymization_method"] == "presidio":
                new_query = self.anonymizer.anonymize(query)
                self.deanonymization_mapping = self.anonymizer.deanonymizer_mapping
                

                
            elif self.guardrail_params["anonymization_method"] == "llm":
                anonymization_dict = self.chatbot.custom_prompt(
                    system_prompt_templates["guardrails"]["anonymization"],
                    user_prompt_templates["guardrails"]["anonymization"].substitute(query=query)
                )
                print(anonymization_dict)

                
                try:
                    anonymization_dict = json.loads(anonymization_dict)
                    self.deanonymization_mapping = {}
                    for key in anonymization_dict:
                        new_query = new_query.replace(key, anonymization_dict[key])
                        self.deanonymization_mapping[anonymization_dict[key]] = key
                    
                except json.JSONDecodeError:
                    pass

            if new_query != query:
                self.logger.log("The LLM guardrail has anonymized the user query")
                self.logger.log(f"[NEW QUERY] {new_query}")

                return GuardrailResponse.CHANGED, new_query, None

        return GuardrailResponse.OK, query, None

    def guardrail_output(self, guardrail_response, result):
        """
        Apply output guardrails to the LLM result.

        Args:
            guardrail_response (GuardrailResponse): The response from the input guardrail.
            result (str): The raw output from the LLM.

        Returns:
            Tuple[GuardrailResponse, str]: A tuple containing:
                - GuardrailResponse: Final status after applying output guardrails.
                - str: Processed result or error message if applicable.
        """

        if guardrail_response == GuardrailResponse.CHANGED:

            if self.guardrail_params["deanonymization_method"] == "combined_exact_fuzzy":
                new_result = self.anonymizer.deanonymize(
                    result,
                    deanonymizer_matching_strategy=combined_exact_fuzzy_matching_strategy
                )

            elif self.guardrail_params["deanonymization_method"] == "llm":
                print(system_prompt_templates["guardrails"]["de-anonymization"])
                print(user_prompt_templates["guardrails"]["de-anonymization"].substitute(anonymized_text=result, mappings=self.deanonymization_mapping))
            
                new_result = self.chatbot.custom_prompt(
                    system_prompt_templates["guardrails"]["de-anonymization"],
                    user_prompt_templates["guardrails"]["de-anonymization"].substitute(anonymized_text=result, mappings=self.deanonymization_mapping)
                )
            else:
                raise Exception("de-anonymization method not known")
            
            self.anonymizer.reset_deanonymizer_mapping()
            self.logger.log("The guardrail has deanonymized the response")
            self.logger.log(f"[New Response] {new_result}")
            return GuardrailResponse.OK, new_result

        return GuardrailResponse.OK, result
