from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
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

loader = TextLoader("text_doc.txt", encoding="utf-8")

docs = loader.load()

prompt = PromptTemplate(
    template='Generate a summary on the following text: \n {text}.',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(result)

