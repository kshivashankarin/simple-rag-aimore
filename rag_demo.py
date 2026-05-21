from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time

# ======================================
# Load Environment Variables
# ======================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ======================================
# OpenAI Client
# ======================================
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================
# Pinecone Client
# ======================================
pc = Pinecone(api_key=PINECONE_API_KEY)

# ======================================
# Create Pinecone Index
# ======================================

print("Pinecone Existing Index -----")
print(pc.list_indexes())
print("+++++++++++++++++++++++++++++")

existing_indexes = [index["name"] for index in pc.list_indexes()]

print("Existing Index -----")
print(existing_indexes)
print("+++++++++++++++++++++++++++++")

if PINECONE_INDEX_NAME not in existing_indexes:
    print("Creating Pinecone index...")

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# ======================================
# Read Documents From Folder
# ======================================
DOCUMENTS_FOLDER = "documents"

all_documents = []

print("Documents Folder files in loop............ ")

for file_name in os.listdir(DOCUMENTS_FOLDER):

    file_path = os.path.join(DOCUMENTS_FOLDER, file_name)

    print("File name : " + file_name + " |  File Path : " + file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

        all_documents.append(
            {
                "id": file_name,
                "text": text
            }
        )

print(f"Loaded {len(all_documents)} documents")

# ======================================
# Create Embedding Function
# ======================================
def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# ======================================
# Convert Documents To Vectors
# ======================================
print("Creating embeddings...")

vectors = []

for doc in all_documents:
    embedding = create_embedding(doc["text"])

    # print("Embedding.........Start ")
    # print(embedding)
    # print("Embedding.........End ")

    vectors.append(
        {
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                "text": doc["text"]
            }
        }
    )


# ======================================
# Upload To Pinecone
# ======================================
index.upsert(vectors=vectors)

print("Documents uploaded successfully!\n")

# ======================================
# Chat Loop
# ======================================
print("========== SIMPLE RAG ==========")
print("Type 'exit' to quit")
print("================================\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        break

    # ======================================
    # Query Embedding
    # ======================================
    query_embedding = create_embedding(user_query)

    # ======================================
    # Search Similar Documents
    # ======================================
    results = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
    )

    # ======================================
    # Extract Context
    # ======================================
    contexts = []

    for match in results["matches"]:
        contexts.append(match["metadata"]["text"])

    final_context = "\n\n".join(contexts)

    print("\nRetrieved Context:\n")
    print(final_context)

    # ======================================
    # Final Prompt
    # ======================================
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

Context:
{final_context}

Question:
{user_query}
"""

    # ======================================
    # Generate AI Response
    # ======================================
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    print("\nAI:")
    print(answer)

    print("\n" + "=" * 50 + "\n")
