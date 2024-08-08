from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from rag.models.databases import VectorDB
from rag.models.guardrails import GuardrailResponse, Guardrails
from rag.models.generation import Generation
from rag.models.retrieval import Retrieval
from rag.models.routing import Routing
from rag.functions.logger import CustomLogger

 
class PipelineState(TypedDict):
    """
    Represents the pipeline state.

    Attributes:
        query: question query
        conversation: full llm conversation
        retrieved_documents: output of retrieval

        result: output of generation
    """
    query: str
    guardrail_response: GuardrailResponse
    conversation: List[str]
    retrieved_documents: List[Document]
    result: str


class Pipeline:
    """
    The main pipeline
    """

    def __init__(self, vectordb: VectorDB, config):
        # Initialize Modules
        self.guardrails = Guardrails(config)
        self.retrieval = Retrieval(config, vectordb)
        self.generation = Generation(config)
        self.routing = Routing(config)
        self.logger = CustomLogger("[PIPELINE]", config["logging"]["filename"])

        # Set Up Workflow
        workflow = StateGraph(PipelineState)
        workflow.add_node("routing", lambda state: state)
        #workflow.add_node("hallucination_detection", lambda state: state)
        #workflow.add_node("guardrail_output_routing", lambda state: state)
        #workflow.add_node("translate_query", self.translate_query_forward)
        workflow.add_node("retrieval", self.retrieval_forward)
        # workflow.add_node("document_relevance", self.document_relevance_forward)
        #workflow.add_node("transform_query", self.transform_query_forward)
        workflow.add_node("generation_rag", self.generate_rag_forward)
        workflow.add_node("generation_no_rag", self.generate_no_rag_forward)
        workflow.add_node("guardrail_input_check", self.guardrail_input_check)
        workflow.add_node("guardrail_output_check", self.guardrail_output_check)

        # Build graph
        workflow.set_entry_point("guardrail_input_check")

        workflow.add_conditional_edges(
            "guardrail_input_check",
            self.guardrail_input_routing,
            {
                "ok": "routing",
                "not_ok": END
            }
        )

        workflow.add_conditional_edges(
            "routing",
            self.rag_relevance_decision,
            {
                "rag": "retrieval",
                "no_rag": "generation_no_rag"
            }
        )
        # No RAG
        workflow.add_edge("generation_no_rag", "guardrail_output_check")

        # RAG
        # workflow.add_edge("retrieval", "document_relevance")
        # workflow.add_edge("document_relevance", "generation_rag")
        workflow.add_edge("retrieval", "generation_rag")
        workflow.add_edge("generation_rag", "guardrail_output_check")
        
        # Output Guardrail
        workflow.add_edge("guardrail_output_check", END)
        self.app = workflow.compile()


        ######## END of Set Up ########
        self.logger.log("Pipeline Initialized")

    """
    Build Pipeline Nodes
    """

    def invoke(self, query: str, conversation: List[str]):
        """
        Invokes the pipeline with a query and returns the resulting response and used documents

        Params:
            query: the query
            conversation: the previous conversation
        Returns:
            the response and used documents
        """
        
        self.logger.log("---------- New Request ----------")
        self.logger.log(f"[USER QUERY] {query}")

        last_state = self.app.invoke(
            {"query": query,
             "conversation": conversation,
             "retrieved_documents": [],
             "result": "",
             "guardrail_response": None
             }
        )
        return last_state['result'], last_state["retrieved_documents"]

    def retrieve(self, query: str, conversation: List[str]) -> dict:
        """
        Invokes the pipeline with a query and returns the full end state

        Params:
            query: the query
            conversation: the previous conversation
        Returns:
            the end state of the pipeline
        """
        last_state = self.app.invoke(
            {"query": query,
             "conversation": conversation,
             "retrieved_documents": [],
             "result": "",
             }
        )
        return last_state

    """
    GUARDRAILS
    """

    def guardrail_input_check(self, state: PipelineState) -> PipelineState:
        query = state['query']
        guardrail_response, new_query, error_message = self.guardrails.guardrail_input(query)
        new_state = state.copy()

        new_state["guardrail_response"] = guardrail_response
        new_state["query"] = new_query
        new_state["result"] = error_message

        return new_state

    def guardrail_input_routing(self, state: PipelineState) -> str:
        if (state["guardrail_response"] == GuardrailResponse.OK
                or state["guardrail_response"] == GuardrailResponse.CHANGED):
            return "ok"
        elif state["guardrail_response"] == GuardrailResponse.NOT_OK:
            return "not_ok"
        else:
            raise Exception("The guardrail routing cannot be called without setting the guardrail_response")

    def guardrail_output_check(self, state: PipelineState) -> PipelineState:
        guardrail_response, result = self.guardrails.guardrail_output(state["guardrail_response"], state['result'])
        new_state = state.copy()

        new_state["guardrail_response"] = guardrail_response
        new_state["result"] = result

        return new_state

    def guardrail_output_routing(self, state):
        if (state["guardrail_response"] == GuardrailResponse.OK
                or state["guardrail_response"] == GuardrailResponse.CHANGED):
            return "ok"
        elif state["guardrail_response"] == GuardrailResponse.NOT_OK:
            return "not_ok"
        else:
            raise Exception("The guardrail routing cannot be called without setting the guardrail_response")

    """
    WORKFLOW NODES & EDGES
    """

    def translate_query_forward(self, state: PipelineState) -> PipelineState:
        """
        Translation query node.

        Params:
            state: current state
        Returns:
            new state
        """
        query = state['query']
        translated_query = self.routing.translate_query(query)
        new_state = state.copy()
        new_state['query'] = translated_query
        return new_state

    def document_relevance_forward(self, state: PipelineState) -> PipelineState:
        """
        Document relevance node.

        Params:
            state: current state
        Returns:
            new state
        """
        if self.routing.document_relevance(state['query'], state['retrieved_documents']):
            return state
        else:
            new_state = state.copy()
            new_state['retrieved_documents'] = []
            return new_state

    def transform_query_forward(self, state: PipelineState) -> PipelineState:
        """
        Transform query node.

        Params:
            state: current state
        Returns:
            new state
        """
        # TODO: This is already done in chatbot.py, if hallcuination detection is deleted this node should also be deleted

        query = """You're only allowed to respond with: "I'm sorry, but there were no documents found that 
        match your query. Please consider reformulating your question or feel free to ask about a different topic." """

        new_state = state.copy()
        new_state['query'] = query
        new_state['retrieved_documents'] = []
        return new_state

    def generate_rag_forward(self, state: PipelineState) -> PipelineState:
        result = self.generation.generate_rag(state['query'], state['conversation'], state['retrieved_documents'])
        new_state = state.copy()
        new_state['result'] = result
        return new_state
    
    def retrieval_forward(self, state: PipelineState) -> PipelineState:
        """
        Retrieval node.

        Params:
            state: current state
        Returns:
            new state
        """
        query = state['query']
        retrieved_documents = self.retrieval.retrieve_documents(query)
        new_state = state.copy()
        new_state['retrieved_documents'] = retrieved_documents
        return new_state

    def generate_no_rag_forward(self, state: PipelineState) -> PipelineState:
        result = self.generation.generate_no_rag(state['query'], state['conversation'])
        new_state = state.copy()
        new_state['result'] = result
        return new_state

    def rag_relevance_decision(self, state: PipelineState) -> str:
        """
        Returns a decision whether RAG is needed for the query or not

        Params:
            state: current state
        Returns:
            new state
        """
        return self.routing.rag_relevance(state['query'])


    def hallucination_detection_decision(self, state: PipelineState) -> str:
        """
        Returns a decision whether the chatbot hallucinated

        Params:
            state: current state
        Returns:
            new state
        """
        return self.routing.hallucination_detection(state['query'], state['retrieved_documents'], state['result'])
