from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

embedding_function = OpenAIEmbeddings()

db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')

combine_prompt = PromptTemplate.from_template(
    template="""Write a summary of the following text. \n\n{summaries}"""
)

question_prompt= PromptTemplate.from_template(
    template="""Use the following piece of context to answer the question at the end. If you do not know the answer, just say you do not know and do not make up an answer. 
    {context}
    Question: {question}
    Helpful answer:
    """
)


# retriever = db.as_retriever()
# doc_num=3
# response = db.max_marginal_relevance_search(query="Tell me a name of a fragrance that has citrus notes", k=doc_num) # showing 2 relevant document

# for i in range(0, doc_num):
#     print(response[i])
#     print('\n')

# 1 way
# QA_prompt = PromptTemplate(
#     template="""Use the following pieces of context to answer the user question. 
#     context: {text}
#     question: {quesion}
#     Answer:""",
#     input_variables=["text", "question"]
# )

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(search_kwargs={"fetch_k": 3, "k": 3}, search_type='mmr'),
    return_source_documents=True,
    chain_type="map_reduce", # mapreduce; it's not as good when you're finding a bunch of things
    chain_type_kwargs={"question_prompt": question_prompt, "combine_prompt": combine_prompt}
)

question = "What areas are the least commonly used ingredients under the notes? "

response = qa_chain({"query": question})

print(response['result'])
print('========')
print(response['source_documents'])



# db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')
# document_content_description="A fragrance database that contains information on the name, brand, and the notes of the scent. "

# llm = OpenAI(temperature=0)
# retriever = SelfQueryRetriever.from_llm(llm,
#     vectorstore=db, 
#     document_contents=document_content_description,
#     verbose=True)


# metadata_field_info = [
#     AttributeInfo(
#         name='page_content',
#         description='This is the name and the brand of the perfume',
#         type='string'
#     )
# ]

# doc_num=3

# docs = docs.get_relevant_documents("What are fragrances that contain Yuzu?", k=doc_num)

# for i in range(0, doc_num):
#     print(docs[i])
#     print('\n')