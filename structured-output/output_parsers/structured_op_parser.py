from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    ResponseSchema(name='fact_4', description='Fact 4 about the topic'),
    ResponseSchema(name='fact_5', description='Fact 5 about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 5 facts about the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'Black holes'})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)

