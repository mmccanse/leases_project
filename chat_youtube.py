import os
import streamlit as st
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import YoutubeLoader


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        
st.title('Youtube Chat Assistant')
youtube_url = st.text_input('Input your Youtube URL')
question = st.text_area('Input your question')
add_file = st.button('Submit', on_click=clear_history)

if youtube_url and add_file:
    with st.spinner('Reading, chunking, and embedding file...'):
        
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=.4)
        
        retriever = vector_store.as_retriever()
        
        crc = ConversationalRetrievalChain.from_llm(llm,retriever)
        st.session_state.crc = crc
        st.success('File uploaded, chunked and embedded successfully')
        


if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        if 'history' not in st.session_state:
            st.session_state['history'] = []
            
        response = crc.run({'question':question, 'chat_history':st.session_state['history']})
        
        st.session_state['history'].append((question,response))
        st.write(response)
        st.divider()
        
        st.write('Conversation History')
        for prompts in st.session_state['history']:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])
            
