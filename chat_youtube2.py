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
process_video = st.button('Process video')

if process_video and youtube_url:
    with st.spinner('Reading, chunking, and embedding...'):
        
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
        st.success('Video processed and ready for queries')
        
question = st.text_area('Input your question')
submit_question = st.button('Submit question')
clear_history_button = st.button('Clear History')
if clear_history_button:
    clear_history()

if submit_question and question and 'crc' in st.session_state:
    crc = st.session_state.crc
    if 'history' not in st.session_state:
        st.session_state['history'] = []
            
    response = crc.run({'question':question, 'chat_history':st.session_state['history']})
        
    st.session_state['history'].append((question,response))
    st.write(response)
    st.divider()
        
    st.markdown(f"**Conversation History**")
    for prompts in st.session_state['history']:
        st.markdown(f"**Question:** {prompts[0]}")
        st.markdown(f"**Answer:** {prompts[1]}")
            
