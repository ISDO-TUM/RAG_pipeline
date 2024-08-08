import re
import sys
from abc import ABC, abstractmethod
from string import Template
from typing import List
import requests
from langchain_core.documents import Document
from openai import OpenAI


class Chatbot(ABC):
    """
    A LLM chatbot that can answer questions
    """
    @abstractmethod
    def answer_question(self, query: str) -> str:
        """
        Answer a question given a query

        Params:
            query: the question
            conversation: the previous conversation
            documents: documents accompanying the query
        Returns:
            the chatbot answer
        """
        pass

    @abstractmethod
    def custom_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
        """
        Answer a question given a query, manually configuring the system and user prompt

        Params:
            system_prompt: the system prompt
            user_prompt: the user prompt
            temperature: the llm temperature
        Returns:
            the chatbot answer
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        the specific model name of the chatbot

        Returns:
            the specific model name
        """
        pass


class TestChatbot(Chatbot):
    """
    A chatbot for testing purposes. Always answers with a fake response.
    """
    def __init__(self, fake_model_name, fake_response="Test"):
        self.fake_model_name = fake_model_name
        self.fake_response = fake_response

    def answer_question(self, query: str, conversation: List[str], documents: List[Document]) -> str:
        return self.fake_response

    def custom_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
        return self.fake_response

    @property
    def model_name(self) -> str:
        return self.fake_model_name


def format_source(response: str) -> str:
    """
    Replaces all sources in a chatbot answer with a html link

    Params:
        response: the chatbot response containing sources
    Returns:
        formatted response
    """
    pattern1 = r'\[([^\]]+)\]\(([^\)]+)\)'

    def replacer(match):
        link_text = match.group(1)
        url = match.group(2)
        return f' <a href="{url}">{link_text}</a>'

    formatted_response = re.sub(pattern1, replacer, response)

    return formatted_response


def calculate_cost(completion_tokens: int, prompt_tokens: int, model: str) -> None:
    pricing = {
        'gpt-3.5-turbo': {
            'prompt': 1.5,
            'completion': 2.0,
        },
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("TODO Invalid model specified")

    prompt_cost = prompt_tokens * model_pricing['prompt'] / 1000000
    completion_cost = completion_tokens * model_pricing['completion'] / 1000000

    total_cost = prompt_cost + completion_cost
    print(f"[COST] Tokens used:  {prompt_tokens} prompt + {completion_tokens} completion = {completion_tokens + prompt_tokens} tokens")
    print(f"[COST] Total cost for {model}: ${total_cost:.4f}")


class OpenAIChatbot(Chatbot):
    def __init__(self, api_key, model, default_system_prompt=None):
        self.openai_client = OpenAI(api_key=api_key)
        self.model = model
        self.default_system_prompt = default_system_prompt

    def answer_question(self, question):
        answer = self.custom_prompt(self.default_system_prompt, question)
        formatted_answer = format_source(answer)

        return formatted_answer

    def custom_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0.4):
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )

        return completion.choices[0].message.content

    @property
    def model_name(self) -> str:
        return "openai"


class OllamaChatbot(Chatbot):
    """
    Ollama chatbot
    """
    def __init__(self, url, model, default_system_prompt=None):
        self.url = url
        self.model = model
        self.default_system_prompt = default_system_prompt


    def answer_question(self, question):
        answer = self.custom_prompt(self.default_system_prompt, question)
        formatted_answer = format_source(answer)

        return formatted_answer

    def custom_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
        """
        Prompts the Ollama chatbot with custom parameters

        Args:
            system_prompt: the system prompt
            user_prompt: the user prompt
            temperature: the temperature. Currently not implemented in this chatbot.
        Returns:
            the llm response
        """
        payload = {
            "model": self.model,
            "prompt": system_prompt + user_prompt,
            "stream": False
        }
        response = requests.post(self.url, json=payload)
        return response.json()["response"]

    @property
    def model_name(self) -> str:
        return self.model


def get_chatbot(config, provider, model, default_system_prompt=None) -> "Chatbot":
    """
    Returns a chatbot configured according to the given parameters

    Params:
        config: the config file
        provider: the model provider
        model: the specific model
        prompt_template_name: the prompt template
    Returns:
        the configured chatbot
    """
    
    if provider == "openai":
        return OpenAIChatbot(config["chatbot"]["openai_api_key"], model, default_system_prompt)
    elif provider == "ollama":
        return OllamaChatbot(config["chatbot"]["ollama_url"], model, default_system_prompt)
    else:
        print(f"[ERROR] Chatbot Model {model} from {provider} not available.")
        sys.exit(0)