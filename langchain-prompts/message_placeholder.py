from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

chat_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=chat_model)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history_list = []
with open('langchain-prompts/chat_history.txt') as f:
    chat_history_list.extend(f.readlines())

print(chat_history_list)

prompt = chat_template.invoke({'chat_history':chat_history_list, 'query':'Where is my refund ?'})

print(prompt)