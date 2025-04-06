import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === Config ===
PERSIST_DIR = "./db"
CONFIG_FILE = "pdf_config.txt"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("Ask Questions from Your PDFs")

# === Load List of PDFs from Config ===
def load_pdf_list():
    with open(CONFIG_FILE, "r") as f:
        return [line.strip() for line in f if line.strip().endswith(".pdf")]

pdf_files = load_pdf_list()

# === Load or Create Vector Store ===
@st.cache_resource
def load_or_create_vectorstore(pdf_list):
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        query_instruction="Represent the query for retrieval:",
        model_kwargs={"token": HUGGINGFACE_TOKEN}
    )

    # Check if vector DB already exists
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        st.info("🔁 Loaded existing vector DB")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

    # Else, ingest PDFs
    st.info("Creating new vector DB from PDFs...")
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file_path in pdf_list:
        st.write(f"Reading {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vectordb = Chroma.from_documents(all_chunks, embedding, persist_directory=PERSIST_DIR)
    vectordb.persist()
    st.success("Vector DB created and saved!")
    return vectordb

vectordb = load_or_create_vectorstore(pdf_files)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# === Load LLM ===
llm = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 200}
)

# === Custom Prompt ===
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions based on the provided context.
Always answer in a concise and clear manner.
If the answer cannot be found in the context, say "Sorry, I don’t know based on the document."

Context:
{context}

Question: {question}
Answer:
"""
)

# === QA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# === User Query ===
query = st.text_input("💬 Ask a question based on the PDFs...", "")

if query:
    with st.spinner("🤖 Generating answer..."):
        result = qa_chain.invoke(query)

        st.markdown("### ✅ Answer")
        st.write(result["result"])

        st.markdown("---")
        st.markdown("### 📎 Source Documents")
        for i, doc in enumerate(result["source_documents"]):
            source = doc.metadata.get("source", "Unknown source")
            st.markdown(f"**Document {i+1}:** `{source}`")
            st.markdown(doc.page_content[:300] + "...")
