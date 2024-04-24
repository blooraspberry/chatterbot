from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')

# docs = db3.similarity_search(query)
# print(docs[0].page_content)

# SIMILARITY SEARCH - most basic type

retriever = db.as_retriever()

response = db.similarity_search(query="Tell me a name of a fragrance that has citrus notes", k=2) # showing 2 relevant document

print(response[0])

print(response[1])