import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
import os
COHERE_API_KEY = "key"
os.environ["OPENAI_API_KEY"] = "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"


def process_data(file_content, text_input):
    loader = PyPDFLoader(file_content)
    documents = loader.load()

    # Perform your processing logic here
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    persist_directory = 'db'
    embedding = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    qa = VectorDBQA.from_chain_type(llm=Cohere(
        cohere_api_key=COHERE_API_KEY), chain_type="stuff", vectorstore=vectordb)
    result = qa.run(text_input)
    print(result)
    processed_text = process_text(result)
    st.text_area("Processed Text:", value=processed_text)
    return result


def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as file:
        file.write(uploaded_file.getbuffer())
    st.success("wait")


def process_text(input_text):
    # Perform your text processing logic here

    processed_text = input_text.upper()
    return processed_text


def main():
    st.title("77AI Governance AI")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    # Text Input BoxPyPDFLoader
    text_input = st.text_input("Enter  yout query")

    #  Button
    if st.button("Run!") and uploaded_file is not None:
        save_uploaded_file(uploaded_file)
        print(uploaded_file.name)
        processed_data = process_data(uploaded_file.name, text_input)


if __name__ == "__main__":
    main()
