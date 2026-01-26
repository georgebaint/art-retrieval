import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="my_collection"
)

# Add documents with embeddings
collection.add(
    ids=["id1", "id2"],
    documents=["This is a document", "Another doc"],
    embeddings=[[1.2, 2.3], [3.4, 4.5]]
)

# Query by vector similarity
results = collection.query(
    query_embeddings=[[1.1, 2.2]],
    n_results=1
)

print(results)
