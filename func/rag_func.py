import chardet

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
    return detected_encoding

def load_data_from_csv(file_path):
    encoding = detect_encoding(file_path)
    print(f"Detected RAG data encoding: {encoding}")
    loader = CSVLoader(
        file_path,
        encoding = encoding)
    return loader.load()

def split_documents(documents, splitter_type = 'Semantic'):
    embedding_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-mpnet-base-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )

    if splitter_type.lower() == 'semantic':
        splitter = SemanticChunker(embedding_model,
                                   breakpoint_threshold_type = 'gradient',
                                   breakpoint_threshold_amount = 0.8)
    elif splitter_type.lower() == 'recursive':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
    else:
        raise ValueError(
            "Invalid splitter type. Current available options are 'Semantic' or 'Recursive'"
            )

    return splitter.split_documents(documents), embedding_model

def create_retriever(docs_chunks, embedding_model, num_k = 3):
    vector_store = Chroma.from_documents(
        documents = docs_chunks,
        embedding = embedding_model,
        )
    
    retriever = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': num_k}
    )

    return retriever, vector_store