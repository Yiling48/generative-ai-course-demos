import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def load_documents(folder_path):
    documents = []

    for path in Path(folder_path).iterdir():
        if path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".docx":
            loader = UnstructuredWordDocumentLoader(str(path))
        else:
            continue

        documents.extend(loader.load())

    return documents


def build_vector_db(
    input_dir="uploaded_docs",
    output_dir="faiss_db",
    chunk_size=500,
    chunk_overlap=100,
):
    os.makedirs(input_dir, exist_ok=True)

    documents = load_documents(input_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {input_dir!r}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)

    embeddings = CustomE5Embedding(
        model_name="intfloat/multilingual-e5-small"
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(output_dir)

    return len(documents), len(split_docs)


if __name__ == "__main__":
    n_docs, n_chunks = build_vector_db()
    print(f"Built FAISS DB from {n_docs} documents / {n_chunks} chunks.")
