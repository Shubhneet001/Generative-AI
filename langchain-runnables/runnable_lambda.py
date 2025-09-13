from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

# def word_count(text: str) -> int:
#     return len(text.split())

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

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cats'}))