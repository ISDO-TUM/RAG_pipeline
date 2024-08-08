from typing import List

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rag.functions.logger import CustomLogger
from rag.models.chatbot import get_chatbot
from rag.models.databases import VectorDB


class Retrieval:
    def __init__(self, config, vector_db: VectorDB) -> None:
        self.vector_db = vector_db
        self.chatbot = get_chatbot(config, config["retrieval"]["provider"], config["retrieval"]["model"], None)
        self.retrieval_params = config["retrieval"]

        self.logger = CustomLogger("[RETRIEVAL]", config["logging"]["filename"])
        self.logger.log("Retrieval initialised")

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        This method returns the relevant documents based on the approach specified in the config.

        Args:
            query (str): The query string.

        Returns:
            List[Document]: A list of Document objects.
        """
        k = int(self.retrieval_params["k_chunks"])

        if self.retrieval_params["method"] == "Nearest Neighbor":
            retrieved_documents = self.get_top_k_relevant_documents_nearest_neighbor(
                query=query,
                k=k
            )
        elif self.retrieval_params["method"] == "Contextual Compression":
            retrieved_documents = self.get_top_k_relevant_documents_contextual_compression(
                query=query,
                k=k
            )
        elif self.retrieval_params["method"] == "Parent Document":
            retrieved_documents = self.get_top_k_relevant_documents_parent_document(
                query=query,
                k=k
            )
        elif self.retrieval_params["method"] == "SVM":
            retrieved_documents = self.get_top_k_relevant_documents_svm(
                query=query,
                k=k
            )
        elif self.retrieval_params["method"] == "Multi-Query":
            retrieved_documents = self.get_top_k_relevant_documents_multi_query(
                query=query,
                k=k
            )
        elif self.retrieval_params["method"] == "Ensemble":
            retrieved_documents = self.get_top_k_relevant_documents_emsemble(
                query=query,
                k=k
            )
        else:
            self.logger.log(f"[ERROR] Retrieval Method {self.retrieval_params['method']} not available.")
            exit()
        self.logger.log(f"[CONFIG] Retrieval Method {self.retrieval_params['method']}.")

        if self.retrieval_params["reranker"] == "0":
            self.logger.log(f"[CONFIG] No reranking")
        elif self.retrieval_params["reranker"] == "1":
            self.logger.log(f"[CONFIG] Reordering documents with long context reorder")
            retrieved_documents = self.long_context_reorder(documents=retrieved_documents)
        elif self.retrieval_params["reranker"] == "2":
            self.logger.log(f"[CONFIG] Reranking documents with cohere")
            retrieved_documents = self.cohere_reranking(query=query, k=k)
        elif self.retrieval_params["reranker"] == "3":
            self.logger.log(f"[CONFIG] Reranking documents with cross encoder")
            retrieved_documents = self.crossencoder_reranking(query=query, k=k)
        else:
            self.logger.log(f"[ERROR] Reranking Method {self.retrieval_params['reranker']} not available.")
            exit()

        return retrieved_documents

    def get_top_k_relevant_documents_nearest_neighbor(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the nearest neighbor search.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.

        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_base_retriever(k=k)
        docs = retriever.invoke(query)
        return docs

    def get_top_k_relevant_documents_contextual_compression(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the contextual compression of the query.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.

        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_compression_retriever(llm=self.chatbot, k=k)
        docs = retriever.invoke(query)
        return docs

    def get_top_k_relevant_documents_parent_document(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the parent document retriever.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.
        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_parent_document_retriever(chunk_size=400, chunk_overlap=20)
        docs = retriever.get_relevant_documents(query)[:k]
        return docs

    def get_top_k_relevant_documents_svm(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the SVM classifier.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.
        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_svm_retriever()
        docs = retriever.invoke(query)[:k]
        return docs

    def get_top_k_relevant_documents_multi_query(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the multi query retriever.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.
        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_multi_query_retriever(llm=self.chatbot, k=k)
        docs = retriever.invoke(query)
        return docs

    def get_top_k_relevant_documents_emsemble(self, query: str, k=5) -> List[Document]:
        """
        This method returns the top k relevant documents based on the ensemble retriever.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to return.
        Returns:
            List[Document]: A list of Document objects representing the top k relevant documents.
        """
        retriever = self.vector_db.get_ensemble_retriever(weights=[0.5, 0.5], k=k)
        docs = retriever.invoke(query)
        return docs

    # reordering documents
    def long_context_reorder(self, documents: List[Document]) -> List[Document]:
        """
        Main idea behind this approach is that when multiple documents 10+ are retrieved, performance
        degrades when models have to access information in the middle of long contexts, long-context reorder
        reorders the most relevant documents to the start and end. https://arxiv.org/abs//2307.03172
        Args:
            documents (List[Document]): A list of documents

        Returns:
            List[Document]: A list of reordered documents
        """
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(documents)
        return reordered_docs

    #reranking documents using Cohere
    def cohere_reranking(self, query:str , k=5) -> List[Document]:
        retriever= self.vector_db.cohere_compression(k=k)
        return retriever.invoke(query)
    
    def crossencoder_reranking(self, query:str , k=5) -> List[Document]:
        retriever= self.vector_db.cross_encoder_compression(k=k)
        return retriever.invoke(query)
