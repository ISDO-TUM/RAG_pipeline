from abc import ABC

from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever, SVMRetriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class DB(ABC):
    def __init__(self, documents):
        self.documents = documents


class VectorDB(DB, ABC):
    def __init__(self, documents, embedding_function):
        super().__init__(documents)
        self.embedding_function = embedding_function
        self.vector_db = None
        self.retriever = None

    def get_base_retriever(self, k):
        try:
            if not self.retriever:
                self.retriever = self.vector_db.as_retriever(search_type="similarity_score_threshold",
                                                             search_kwargs={"score_threshold": 0.4, "k": k})
            return self.retriever
        except AttributeError:
            raise NotImplementedError("The vector database has not been initialized for this instance of VectorDB.")

    def get_compression_retriever(self, llm, k):
        if not self.retriever:
            retriever = self.get_base_retriever(k=k)
            _filter = LLMChainExtractor.from_llm(llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
        return self.retriever
    
    def get_parent_document_retriever(self, chunk_size, chunk_overlap):
        if not self.retriever:
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            # vector store for child chunks 
            vectorstore = Chroma(embedding_function=self.embedding_function)
            # store for parent documents without overhead of embedding costs
            store = InMemoryStore()

            parent_document_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter
            )
            parent_document_retriever.add_documents(self.documents)
            self.retriever = parent_document_retriever
        return self.retriever
    
    def get_svm_retriever(self):
        if not self.retriever:
            self.retriever = SVMRetriever.from_documents(self.documents, self.embedding_function)
        return self.retriever

    def get_multi_query_retriever(self, llm, k):
        if not self.retriever:
            self.retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=self.get_base_retriever(k=k))
        return self.retriever

    def get_ensemble_retriever(self, weights, k):
        if not self.retriever:
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = k
            self.retriever = EnsembleRetriever(retrievers=[bm25_retriever, self.get_base_retriever(k=k)],
                                               weights=weights)
        return self.retriever

    def cohere_compression(self, k):
        retriever = self.get_base_retriever(k=k)
        compressor = CohereRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def cross_encoder_compression(self,k):
        retriever = self.get_base_retriever(k=k)

        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model)

        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever

    def __str__(self) -> str:
        return f"VectorDB with {len(self.documents)} documents and {self.embedding_function} as embedding function"


class ChromaDB(VectorDB):
    def __init__(self, embedding_function, documents=None, chroma=None, retriever=None, persist_current_vectordb=False, use_persist_directory=False, persist_directory=None):
        super().__init__(documents, embedding_function)
        self.retriever = retriever
        
        if use_persist_directory:
            self.vector_db = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)
            print("[INFO] Using persist directory")
        elif chroma:
            self.vector_db = chroma
        else:
            if persist_current_vectordb: 
                self.vector_db = Chroma.from_documents(documents=documents, embedding=embedding_function, persist_directory=persist_directory)
                print("[INFO] Persist directory created")
            else: 
                self.vector_db = Chroma.from_documents(documents=documents, embedding=embedding_function)
                

class FaissDB(VectorDB):
    def __init__(self, documents, embedding_function):
        super().__init__(documents, embedding_function)
        self.vector_db = FAISS.from_documents(documents=documents, embedding=embedding_function)
