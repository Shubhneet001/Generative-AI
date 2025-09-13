from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
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
    template='Generate a tweet about {topic}',
    input_variables=['topic'],
)
prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic'],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'linkedin_post':RunnableSequence(prompt2, model, parser)
})


result = parallel_chain.invoke({'topic':'cats'})
print(result)