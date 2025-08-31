from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = input("Query: ")

documents_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], documents_embedding)[0]
index, score = sorted(list(enumerate(similarity_scores)), key=lambda x:x[1])[-1]

# print("Query: ", query)
print("Response: ", documents[index])
print("Similarity score is: ", score)




