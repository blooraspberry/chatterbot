from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma

embedding_function = OpenAIEmbeddings()



db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')


llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=db.as_retriever()
)
doc_num=3
compressed_docs = compression_retriever.get_relevant_documents("What are the name and brand of fragrances that contain jasmine?", k=doc_num)


# docs = db3.similarity_search(query)
# print(docs[0].page_content)

# SIMILARITY SEARCH - most basic type

# retriever = db.as_retriever()

# response = db.max_marginal_relevance_search(query="Tell me a name of a fragrance that has citrus notes", k=doc_num) # showing 2 relevant document

for i in range(0, doc_num):
    print(compressed_docs[i])
    print('\n')