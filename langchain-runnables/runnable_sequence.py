from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    temperature=2.0
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Explain the follwing joke: {joke}',
    input_variables=['joke'],
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'cats'}))