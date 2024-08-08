import uuid
from typing import List, Dict

import dotenv
import tiktoken
import tqdm
from langchain.retrievers import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.stores import InMemoryByteStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TextSplitter

from rag.fixtures.prompts import system_prompt_templates
from rag.models.chatbot import Chatbot, get_chatbot
from rag.models.databases import ChromaDB, VectorDB
from rag.models.dataloader import DataLoader

dotenv.load_dotenv()


def calculate_embedding_cost(documents: List[Document], model: str) -> None:
    """Computes embedding cost for given embedding model

    Args:
        documents: the langchain documents to embedd
        model: the embedding model
    Returns:
        None
    """

    embedding_cost = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "ada v2": 0.10
    }

    if model in embedding_cost.keys():
        n_tokens = 0
        print("[INFO] Computing cost for embedding.")
        encoding = tiktoken.encoding_for_model(model)
        for doc in tqdm.tqdm(documents):
            n_tokens += len(encoding.encode(doc.page_content))

        cost = embedding_cost[model] * (n_tokens / 1000000)

        print(f"[COST] {n_tokens} Tokens cost you {cost}$ using {model}.")


def split_documents(documents: List[Document], splitter: TextSplitter) -> List[Document]:
    """Splits langchain documents in chunks

    Documents are split into chunks using the RecursiveCharacterTextSplitter.

    Args:
        documents: the langchain documents to split
        splitter: the langchain textsplitter to use
    Returns:
        split langchain documents
    """
    return splitter.split_documents(documents)


def index_documents(splitter: TextSplitter, embeddings: Embeddings, embeddings_model: str,
                    data_loader: DataLoader, persist_current_vectordb: bool=False,
                    use_persist_directory: bool=False, persist_directory: str=None) -> ChromaDB:
    """Indexes documents and puts them into a chroma vector database

    Therefore, it loads json files, split them into chunks and embeds them into a chroma vector database

    Args:
        splitter: the langchain textsplitter to use
        embeddings: the embeddings to use for the vector database
        embeddings_model: used for calculating the tokens and cost
        data_loader: used to load the data
    Returns:
        the chroma database
    """
    documents = data_loader.load_data()

    calculate_embedding_cost(documents=documents, model=embeddings_model)
    split_docs = split_documents(documents, splitter)
    print("[INFO] Creating database...")
    return ChromaDB(embedding_function=embeddings, documents=split_docs, persist_current_vectordb=persist_current_vectordb,
                    use_persist_directory=use_persist_directory, persist_directory=persist_directory)


def index_documents_with_summaries(splitter: TextSplitter, embeddings: Embeddings, embeddings_model: str,
                                   data_loader: DataLoader, chatbot: Chatbot) -> ChromaDB:
    """Indexes documents and summaries each chunk with the chatbot. Both are put into the database according to the
    parent document architecture.

    Args:
        splitter: the langchain textsplitter to use
        embeddings: the embeddings to use for the vector database
        embeddings_model: used for calculating the tokens and cost
        data_loader: used to load the data
        chatbot: the chatbot to use for summaries
    Returns:
        the chroma database
    """
    documents = data_loader.load_data()

    calculate_embedding_cost(documents=documents, model=embeddings_model)
    split_docs = split_documents(documents, splitter)
    print("[INFO] Creating database with summaries...")

    summaries = []
    for doc in split_docs:
        summaries.append(chatbot.custom_prompt(system_prompt_templates["summary"], doc.page_content))
    vectorstore = Chroma(collection_name="summaries", embedding_function=embeddings)
    store = InMemoryByteStore()
    id_key = "document_id"

    retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)
    doc_ids = [str(uuid.uuid4()) for _ in split_docs]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i], **split_docs[i].metadata})
                    for i, s in enumerate(summaries)]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, split_docs)))
    return ChromaDB(embedding_function=embeddings, documents=documents, chroma=vectorstore, retriever=retriever)


def get_embeddings_and_text_splitter(index_config, openai_api_key: str) -> (Embeddings, TextSplitter):
    """
    Returns the indexing embedding model and text splitter according to the config.

    Params:
        index_config: the indexing configuration
        openai_api_key: the openai api key for the openai embedding models and semantic text splitter
    Returns:
        embedding model and text splitter
    """
    print("[INFO] Creating database.")
    if index_config["embeddings"] == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings()
        print(f"[CONFIG] Embedding model {index_config['embeddings']}.")

    elif index_config["embeddings"] == "text-embedding-3-small":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        print(f"[CONFIG] Embedding model {index_config['embeddings']}.")

    elif index_config["embeddings"] == "text-embedding-3-large":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")
        print(f"[CONFIG] Embedding model {index_config['embeddings']}.")

    elif index_config["embeddings"] == "ada v2":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="ada v2")
        print(f"[CONFIG] Embedding model {index_config['embeddings']}.")

    else:
        print(f"[Error] Embedding model {index_config['embeddings']} not available.")
        exit()
    print(f"[CONFIG] Text Splitter {index_config['textsplitter']}.")
    if index_config["textsplitter"] == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(index_config["textsplitter_recursive_chunk_size"]),
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
    elif index_config["textsplitter"] == "SemanticTextSplitter":
        text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=openai_api_key),
                                        breakpoint_threshold_type=index_config["textsplitter_semantic_breakpoint_type"])
    else:
        print("[Error]  Text Splitter {config['textsplitter']} not available.")
        exit()

    return embeddings, text_splitter


def get_vectordb(config, data_loader: DataLoader) -> VectorDB:
    """
    Builds a vector database by indexing, chunking and embedding all documents.

    Args:
        config: index section from config.ini
    Returns:
        returns a chromadb vector database
    """
    index_config = config["indexing"]
    embeddings, text_splitter = get_embeddings_and_text_splitter(index_config, config["chatbot"]["openai_api_key"])
    
    if index_config["use_summaries"] == "True":
        summary_chatbot = get_chatbot(config, config["indexing"]["provider"], config["indexing"]["model"], None)
        vectordb = index_documents_with_summaries(
            splitter=text_splitter,
            embeddings=embeddings,
            embeddings_model=index_config['embeddings'],
            data_loader=data_loader,
            chatbot=summary_chatbot
        )
    else:
         vectordb = index_documents(
            splitter=text_splitter,
            embeddings=embeddings,
            embeddings_model=index_config['embeddings'],
            data_loader=data_loader,
            persist_current_vectordb=index_config["persist_current_vectordb"] == "True",
            use_persist_directory=index_config["use_persist_directory"] == "True",
            persist_directory=index_config["persist_directory"]
        )

    print("[INFO] Vector Database created.")
    return vectordb
