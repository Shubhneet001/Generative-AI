from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# prompt 1
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# prompt 2
prompt2 = PromptTemplate(
    template='Extract the 10 most important points from the following text: \n {text} \n {format_instructions}',
    input_variables=['text'],
    partial_variables={'format_instructions': "Be very specific about the points, don't include any other text and don't use markdown format."}
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Unemployment in India'})
print(result)

# chain.get_graph().print_ascii()