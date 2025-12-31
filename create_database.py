import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env (if present)
load_dotenv()

# Directory where Chroma will persist its data
CHROMA_PATH = "chroma"
# Local folder containing the source PDFs to ingest
DATA_DIR = r"D:\Godai Tech"

# List of PDFs expected to exist in DATA_DIR
PDF_FILES = [
    "Godai_Tec_Commercial_Terms_and_Pricing_Schedule.pdf",
    "Godai_Tec_Legal_Regulatory_and_Compliance_Framework.pdf",
    "Godai_Tec_Corporate_Policy_and_Operating_Manual.pdf",
]


def main():
    # Entry point: generate the persisted Chroma datastore
    generate_data_store()


def generate_data_store():
    # Load the PDF pages, split them into chunks, and save embeddings to Chroma
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents() -> list[Document]:
    # Ensure the data directory exists
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Folder not found: {DATA_DIR}")

    # Build absolute paths for the expected PDFs
    pdf_paths = [os.path.join(DATA_DIR, f) for f in PDF_FILES]

    # Check for missing files and surface a clear error if any are absent
    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "These PDF files were not found:\n" + "\n".join(missing)
        )

    all_docs: list[Document] = []
    for path in pdf_paths:
        print(f"Loading: {path}")
        # Use PyPDFLoader to extract pages as Document objects
        loader = PyPDFLoader(path)
        docs = loader.load()

        print(f"  Pages loaded: {len(docs)}")

        # Quick sanity check: print length of first-page extracted text
        if docs:
            sample = (docs[0].page_content or "").strip()
            print(f"  First-page text length: {len(sample)}")

        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No pages were loaded from the PDFs.")

    return all_docs


def split_text(documents: list[Document]) -> list[Document]:
    # Use a RecursiveCharacterTextSplitter to break pages into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    db.add_documents(chunks)
    db.persist()

    print(f"Saved {len(chunks)} chunks to ./{CHROMA_PATH}/")

if __name__ == "__main__":
    main()
