from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of any 10 characters in starwars universe. {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Always reply ONLY with valid JSON."),
#     ("user", "Give me the name, age and city of a fictional character in the Star Wars universe. {format_instructions}")
# ]).partial(format_instructions=parser.get_format_instructions())


# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# Build the chain: prompt → model → parser
chain = template | model | parser

# Run it
result = chain.invoke({})
print(result)