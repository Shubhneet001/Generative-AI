from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load Hugging Face API token from .env
load_dotenv()

# Load the document (ensure UTF-8 encoding to avoid decode errors)
loader = TextLoader("graph_NN.txt", encoding="utf-8")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# Create a retriever (fetches relevant documents)
retriever = vector_store.as_retriever()

# Query
query = "What are the main topics that are explained in this blog ?"
retrieved_docs = retriever.invoke(query)   # ✅ updated: .invoke instead of deprecated .get_relevant_documents

# Combine retrieved text into a single prompt
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

# Initialize the chat LLM (Cerebras models only support conversational task)
llm = ChatHuggingFace(
    model="meta-llama/Llama-3.3-70B-Instruct",
    task="conversational",
    temperature=0.7
)

# Build chat prompt
prompt_template = ChatPromptTemplate.from_template(
    "Based on the following text, answer the question: {query}\n\nText: {retrieved_text}"
)

# Run chain
chain = prompt_template | llm
answer = chain.invoke({"query": query, "retrieved_text": retrieved_text})

# Print the answer
print(answer.content if hasattr(answer, "content") else answer)
