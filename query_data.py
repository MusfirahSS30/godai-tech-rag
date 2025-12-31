import argparse
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env (e.g., GROQ_API_KEY, GROQ_MODEL)
load_dotenv()

# Directory where Chroma persistence files are stored
CHROMA_PATH = "chroma"

# Prompt instructing the model to answer only from provided context
PROMPT_TEMPLATE = """
You are a careful assistant. Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
"""


def main():
    # Parse CLI argument for the user's query text
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Ensure the Chroma DB exists before attempting searches
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Chroma DB not found at ./{CHROMA_PATH}/. Run create_database.py first."
        )

    # Create an embeddings object (local sentence-transformers model)
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Load the persisted Chroma DB using the same embedding function
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform a similarity search to get top-k relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    # Combine retrieved chunks into a single context string for the LLM
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the chat prompt with the context and the user's question
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )

    # Choose the Groq model from environment (fallback to default)
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # Instantiate the Groq chat model; pass API key from env for authentication
    llm = ChatGroq(
    model=model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

    # Invoke the model and extract the textual content of the response
    response = llm.invoke(prompt).content

    # Collect source metadata (file, page, score) for each retrieved chunk
    sources = []
    for doc, score in results:
        src = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", None)
        sources.append({"source": src, "page": page, "score": float(score)})

    # Print the model's answer and the list of sources to the console
    print("Response:\n", response)
    print("\nSources:")
    for s in sources:
        print(s)


if __name__ == "__main__":
    main()
