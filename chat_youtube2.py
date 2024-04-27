import os
import streamlit as st
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import YoutubeLoader
from streamlit_extras.colored_header import colored_header

# Access open AI key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set up Streamlit page configuration
st.set_page_config(page_title=None,
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)


# Define function to clear history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Define header size/color

def header():
    colored_header(
        label ="YouTube Chat Assistant",
        description = "Find a YouTube video with accurate captions and enter the url below",
        color_name="blue-green-70",   
    )
    # additional styling
    st.write("""
    <style>
    /* Adjust the font size of the header */
    div[data-baseweb="header"] > div {
    font-size: 48px !important;
    }

    /* Adjust the thickness of the line */
    div[data-baseweb="header"] > hr {
    height: 40px !important;
    background-color: rgb(0, 212, 177) !important;
    }

    /* Adjust the font size of the description */
    div[data-baseweb="header"] > div > div {
    font-size: 50px !important;
    }
    </style>

    """, unsafe_allow_html=True)
    

# Define main function
def main():
    header()
    # st.title('YouTube Chat Assistant!')
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
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=.2)
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
        for prompts in reversed(st.session_state['history']):
            st.markdown(f"**Question:** {prompts[0]}")
            st.markdown(f"**Answer:** {prompts[1]}")
            st.divider()
        
        if st.session_state['history']:
            st.empty()        
if __name__== '__main__':
    main()