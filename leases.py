
# pip install PyMuPDF

# Imports
import streamlit as st
import fitz  # PyMuPDF
import os
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv


# Set the model name for LLM
OPENAI_MODEL = "gpt-3.5-turbo"

# Store API key as a variable
openai_api_key = st.secrets["OPENAI_API_KEY"]

# define function to clear chat history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Define PDF extraction function, handles 1 PDF at a time
def extract_text_from_pdf(file_path):
    with fitz.open(file_path)as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# Define a directory processing function that reads through the folder where PDFs are saved, checks for 
# PDF files and uses extraction function to extract text from each.

def load_pdfs_from_directory(directory):
    # create dictionary to store text from each PDF
    texts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            texts[filename] = text
    return texts
    


def main():
    st.title('ASC 842 AI Assistant')
    question = st.text_input("Ask a question about lease accounting:")
    
    # Load and prepare documents
    pdf_texts = load_pdfs_from_directory('pdfs')
    documents = [text for _, text in pdf_texts.items()]
    
    if question and documents:
        answer = query_openai(documents, question)
        st.write("Answer:", answer)
        
if __name__ == "__main__":
    main()


