from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

DATA_PATH = "../document/final_perfume_data_recovered.csv"
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"] 

# Create loader 

loader = CSVLoader(DATA_PATH)
docs = loader.load()

print(len(docs))


# embedding
embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding_function, persist_directory="../vector_db", collection_name='fragrance')

# make persistant
db.persist() # this creates vector_db which has the vector store 



