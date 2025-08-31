from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=chat_model)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple terms, what is {topic}')

    # SystemMessage(content='You are a helpful {domain} expert.'),
    # HumanMessage(content='Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain': 'Cricket', 'topic': 'dusra'})

print(prompt)