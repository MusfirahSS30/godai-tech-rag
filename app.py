import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables (GROQ_API_KEY, GROQ_MODEL)
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


@st.cache_resource
def get_db():
    # Cache and return a Chroma instance with the local embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


def main():
    # Streamlit page configuration and title
    st.set_page_config(page_title="Godai Tec Chatbot", layout="wide")
    st.title("Godai Tec Chatbot")

    # Ensure the Chroma DB exists before showing the UI
    if not os.path.exists(CHROMA_PATH):
        st.error("Chroma DB not found. Run: python create_database.py")
        st.stop()

    # Load the cached DB
    db = get_db()

    # UI: user input and top-k slider
    query_text = st.text_input("Ask questions about Godai Tec", placeholder="e.g., What are the commercial pricing terms?")
    k = st.slider("Top-K chunks", 2, 8, 4)

    # When the user clicks the button, retrieve and answer
    if st.button("Search & Answer", type="primary") and query_text.strip():
        # Retrieve top-k similar chunks with relevance scores
        results = db.similarity_search_with_relevance_scores(query_text.strip(), k=k)

        # Build context from retrieved chunks and format the prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text,
            question=query_text.strip(),
        )

        # Get Groq model name from env (fallback provided) and pass API key
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        llm = ChatGroq(
    model=model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


        # Call the model while showing a spinner to the user
        with st.spinner("Generating answer..."):
            answer = llm.invoke(prompt).content

        # Display the model answer
        st.subheader("Answer")
        st.write(answer)

        # Show sources (file, page, score) for transparency
        st.subheader("Sources")
        for doc, score in results:
            src = doc.metadata.get("source", "unknown_source")
            page = doc.metadata.get("page", None)
            st.write(
                f"- **{os.path.basename(src)}** | page: **{page}** | score: **{float(score):.3f}**"
            )

        # Allow users to inspect the raw retrieved context
        with st.expander("Show retrieved context"):
            st.text(context_text)


if __name__ == "__main__":
    main()
