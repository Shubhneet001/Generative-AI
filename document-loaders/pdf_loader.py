from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    temperature=2.0
)

model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader("books/after_dark.pdf")

docs = loader.load()

print(len(docs))
print(docs[1].page_content)