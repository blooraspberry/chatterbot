from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import Chroma


embedding_function = OpenAIEmbeddings()



db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')
document_content_description="A fragrance database that contains information on the name, brand, and the notes of the scent. "

llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(llm,
    vectorstore=db, 
    document_contents=document_content_description,
    verbose=True)


metadata_field_info = [
    AttributeInfo(
        name='page_content',
        description='This is the name and the brand of the perfume',
        type='string'
    )
]

doc_num=3

docs = docs.get_relevant_documents("What are fragrances that contain Yuzu?", k=doc_num)

for i in range(0, doc_num):
    print(docs[i])
    print('\n')