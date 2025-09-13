from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
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
    template='Write a detailed report about {topic}',
    input_variables=['topic'],
)
prompt2 = PromptTemplate(
    template='Summarize the following text: \n {text}',
    input_variables=['text'],
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)),
    (RunnablePassthrough())
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))