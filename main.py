import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'chroma'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'langchain'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'cohere'])


# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'chroma'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'langchain'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'cohere'])


load_dotenv()
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# Load and process the text
loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

embedding = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory)

vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
qa = VectorDBQA.from_chain_type(llm=Cohere(
    cohere_api_key=COHERE_API_KEY), chain_type="stuff", vectorstore=vectordb)

query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)

# To cleanup, you can delete the collection
vectordb.delete_collection()
vectordb.persist()
