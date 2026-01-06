import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_documents(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    
    loader = DirectoryLoader(
        directory_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs = {"encoding":"utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in the directory {directory_path}.")
    
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=200):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk{i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(chunk.page_content)
            print("-"*50)
    
    return chunks

def create_vectorstore(chunks, persist_dir = "db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("---Creating Vectorstore---")

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_dir,
        collection_metadata = {"hnsw:space": "cosine"}
    )

    print("---Finished creating vector store---")
    print("Vectorstore created and saved to {persist_dir}")

    return vectorstore

def main():
    print("Main Function")
    # 1. Loading the files
    directory_path = "./docs"
    documents = load_documents(directory_path)
    print(f"Loaded {len(documents)} documents.")

    # 2. Splitting documents into chunks
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Create vectorstore
    PERSIST_DIR = "db/chroma_db"
    vectorstore = create_vectorstore(chunks, PERSIST_DIR)
if __name__ == "__main__":
    main()