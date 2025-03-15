import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_vectors")

# Check stored document metadata
docs = collection.get(include=["metadatas"])
print(docs["metadatas"])