from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')

# docs = db3.similarity_search(query)
# print(docs[0].page_content)

# SIMILARITY SEARCH - most basic type

retriever = db.as_retriever()
doc_num=3
response = db.max_marginal_relevance_search(query="Tell me a name of a fragrance that has citrus notes", k=doc_num) # showing 2 relevant document

for i in range(0, doc_num):
    print(response[i])
    print('\n')