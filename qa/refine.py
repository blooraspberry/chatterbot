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


QA_prompt= PromptTemplate.from_template(
    template="""Use the following piece of context to answer the question at the end. If you do not know the answer, just say you do not know and do not make up an answer. 
    Context: {text}
    Question: {question}
    Answer:""",
    input_variables=['text', 'question']
)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(),
    return_source_documents=True,
    chain_type="refine", # mapreduce; it's not as good when you're finding a bunch of things

)

question = "What areas are the least commonly used ingredients under the notes? "

response = qa_chain({"query": question})

print(response['result'])
print('========')
print(response['source_documents'])
