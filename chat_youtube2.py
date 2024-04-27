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
from streamlit_extras.stylable_container import stylable_container

# Access open AI key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Define function to clear history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Styles Setup  #########################################################################

# Define header size/color

def header():
    colored_header(
        label ="YouTube Chat Assistant",
        description = "Find a YouTube video with accurate captions. Enter the url below.",
        color_name='light-blue-40'   
    )
    # additional styling
    st.markdown("""
        <style>
        /* Adjust the font size of the header */
        .st-emotion-cache-10trblm.e1nzilvr1 {
            font-size: 60px !important; /* Change this value to increase or decrease font size
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Adjust the thickness of the line */
        hr {
            height: 16px !important; /* Change this value to increase or decrease line thickness
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Adjust the font size of the description */
        div[data-testid="stCaptionContainer"] p {
            font-size: 20px !important; /* Change this value to increase or decrease font size
        }
        </style>
    """, unsafe_allow_html=True)

# Define button style/formatting
    
def video_button():
    with stylable_container(
        key="video",
        css_styles="""
            button {
                background-color: #74eeff;
                color: #000000;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Submit video")


def question_button():
    with stylable_container(
        key="question",
        css_styles="""
            button {
                background-color: #74eeff;
                color: #000000;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Submit question")

def clear_button():
    with stylable_container(
        key="clear",
        css_styles="""
            button {
                background-color: #74eeff;
                color: #000000;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Clear history")
    
# End styles section ###########################################################################


# Define main function
def main():
    header()
    youtube_url = st.text_input('Input YouTube URL')
    process_video = video_button()

    if process_video and youtube_url:
        with st.spinner('Reading, chunking, and embedding...'):
            
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=300)
            chunks = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(chunks, embeddings)
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=.2)
            retriever = vector_store.as_retriever()
            crc = ConversationalRetrievalChain.from_llm(llm,retriever)
            st.session_state.crc = crc
            st.success('Video processed and ready for queries')
            # # clear the crc from the session state
            # st.session_state.pop('crc', None)
            

    question = st.text_area('Input your question')
    st.markdown("""
        <style>
        /* Adjust the font size of the input labels */
        .st-emotion-cache-ue6h4q p {
        font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    submit_question = question_button()
    clear_history_button = clear_button()
    
    if clear_history_button:
        clear_history()
        # st.session_state.pop('history', None)

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