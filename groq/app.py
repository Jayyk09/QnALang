import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddingsfrom 
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Load the API key from the environment
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY or not os.environ.get("OPENAI_API_KEY"):
    st.error("Missing API keys. Please check your environment variables.")

st.title("GROQ Chat w Llama3")

def vectorEmbedding():
    if "vector" not in st.session_state:
        # create embeddings
        st.session_state.embeddings = OpenAIEmbeddings()  # Consistent naming
        # load documents
        st.session_state.loader = PyPDFDirectoryLoader("./all_docs")
        # load the documents
        st.session_state.docs = st.session_state.loader.load()
        # split the text with a chunk size of 1000 and overlap of 200
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # Correct parameter
        )
        # create the final docs with the text splitter
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        # vector store the embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")


prompt_template=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
user_input = st.text_input("Enter your question from the document:")

if st.button("initialize vector store"):
    vectorEmbedding()
    st.write("Vector store DB is ready")

documents_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

if user_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    st.write("Response time: ", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Show Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------')