import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title('Trying to learn how to use this app')

st.write('Welcome to this app, it\'s gonna get better.')

def chunk_document(document_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_text(document_content)
    return document_chunks

uploaded_file = st.file_uploader("choose a file", type=['txt'])

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    st.success('file successfully uploaded')
    
    chunks = chunk_document(file_content)
    
    st.write(f"Document has been chunked into {len(chunks)} parts.")
    st.write("21st chunk content:")
    st.write(chunks[20])

