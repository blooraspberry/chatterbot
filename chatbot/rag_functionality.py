from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory # to rmember chat
from langchain.chains import ConversationalRetrievalChain
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

import os

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

embedding_function = OpenAIEmbeddings()

vector_db = Chroma(persist_directory="../vector_db", embedding_function=embedding_function, collection_name='fragrance')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)


memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory=memory, 
    retriever = vector_db.as_retriever(
        search_kwargs={"fetch_k": 4, "k": 3}, 
        search_type="mmr"),
    chain_type="refine", # mapreduce; it's not as good when you're finding a bunch of thingsls

)

def rag_func(question: str) -> str: 
    """
    This function takes in user question or prompt and return a response
    :param: question: String value of the question or the prompt from the user
    :returns: String value of the answer to the user question
    """

    reponse = qa_chain({"question": question})
    return reponse.get("answer")

