import configparser

from rag.models.chatbot import get_chatbot
from rag.models.generation_evaluation import GenerationTester

"""
### How to use the Generation Tester ###

The tester has 3 main functions:
1. test(): This function generates the llm responses to the test cases. The results are retured as a dict and also stored as a json.
2. evaluate(): This function evaluates the responses from the LLM. Currently this is done by hand
3. score(): This function takes the evaluation and generates the scores for each llm/prompt

The test cases must have the following structure:
```json
{
    "test_name" : "Name of the Test. For example context_fidelity ",
    "test_description": "Here you describe what the test should test and how it should be evaluated",
    
    "test_cases": [
        {
            "question": "Question to the llm",
            "documents": [{
                "page_content": "Here is a list of supplied documents to the llm ",
                "source": "each document has a source"
            }]
        },
        ....(add test cases here)

"""

if __name__ == "__main__":
    # load chatbot based on config
    config = configparser.ConfigParser()
    config.read('config.ini')

    chatbot = get_chatbot(config, provider='openai', model='gpt-3.5-turbo', prompt_template_name='default')

    gen_tester = GenerationTester([chatbot])

    gen_tester.test_and_score(test_cases_path="test_cases/routing/guardrail_pii.json")
