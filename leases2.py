
# !pip install PyMuPDF
# !pip install -U langchain-community
# !pip install PyPDF2
# !pip install pycryptodome


# Imports
import os
import streamlit as st
import PyPDF2
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
import logging
logging.basicConfig(level=logging.DEBUG)



# Set the model name for LLM
OPENAI_MODEL = "gpt-3.5-turbo"

# Store API key as a variable
openai_api_key = st.secrets["OPENAI_API_KEY"]

# define function to clear chat history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title('ASC 842 AI Assistant')
st.header('This assistant is preloaded with accounting guidance related to ASC 842 Leases under US GAAP.')
st.divider()
st.write('This assistant cannot give specific accounting advice. It is a learning tool and a proof of concept. It is not intended for commercial use. For accounting advice, please consult an appropriate professional.')
st.divider()


# Define text extraction from txt file
def extract_text_from_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Failed to read text from {file_path}: {str(e)}")
        return None




# Define text extraction from PDF file
# def extract_text_from_pdf_file(file_path):
#     texts = []
#     pdf_files = []
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if filename.endswith('.pdf'):
#             try:
#                 with open(file_path, 'rb') as file:
#                     reader = PyPDF2.PdfReader(file)
#                     text = ''
#                     for page in range(len(reader.pages)):
#                         page_obj = reader.pages[page]
#                         text += page_obj.extract_text()
#                     texts.append(text)
#                     pdf_files.append(filename)
#             except Exception as e:
#                 logging.error(f"Failed to read text from {file_path}: {str(e)}")
#                 return None
#     print("PDF files extracted")
#     for pdf in pdf_files:
#         print(pdf)
        
#     return texts

def extract_text_from_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(reader.pages)):
                page_obj = reader.pages[page]
                text += page_obj.extract_text()
            return text
    except Exception as e:
        logging.error(f"Failed to read text from {file_path}: {str(e)}")
        return None

    
#Define function to load text files from a directory:
def load_files_from_directory(directory):
    loaded = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf_file(file_path)
        elif filename.endswith('.txt'):
            text = extract_text_from_text_file(file_path)
        else:
            continue
        loaded.append(text)
    return loaded

    

#Global definitions for text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=OPENAI_MODEL)

def split_and_query_text(crc, text, question):
    #split text into manageable parts
    chunks = text_splitter.split_text(text)
    
    #process each part with the conversational retrieval chain (CRC)
    for chunk in chunks:
        response = crc.run({
            'question': question,
            'chat_history': [{'text': chunk}]
        })
        if response:
            return response
    return "No relevant information found"
            

def setup_vector_store(documents):
    #ensure documents is a list of strings
    if not all(isinstance(doc, str) for doc in documents):
        logging.error("Documents for vector store setup are not all strings.")
    chunks = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='db')
    return vector_store



def create_examples():
    examples = [
        {
            "query": "How do I determine the different lease components?",
            "answer": """Lease components are elements of the arrangement that provide the customer with the right to use an 
            identified asset. An entity should consider the right to use an underlying asset to be a separate lease component 
            if the following 2 conditions are met: 1. Lessee can benefit from the ROU asset either on its own or together with 
            other readily available resources and 2. The ROU is neither highly dependent on; nor highly interrelated with the 
            other ROU(s) in the contract. References: KPMG-leaseshandbook section 4.1, PWC-leasesguide0124, section 2.4, 
            EY-financial-reporting-developments-lease-accounting, section 1.4"""
        },
        {
            "query": "I am a lessor, how do I account for lease modifications?",
            "answer": """Several questions must be answered to determine how to appropriately account for lease modifications for 
            a lessor. Is the modified contract a lease, or does it contain a lease? If yes, does the modification result in a 
            separate contract? If yes, account for two separate contracts: the unmodified original contract, and a separate 
            contract accounted for in the same manner as any other new lease. If the modification does not result in a separate 
            contract, remeasure and reallocate the remaining consideration in the contract, reassess the lease classification at 
            the modification effective date, and account for any initial direct costs, lease incentives, and other paymetns made to 
            or by the lessor. Whether or not the lease classification changes, and how it changes drives the appropriate accounting. 
            References: EY - Financial Reporting Developments: lease accounting, section 5.6, PWC - Leases Guide, section 5.6, 
            KPMG - Leases Handbook, section 7.6"""
        }
    ]

    return examples

def setup_prompt_template(examples):
    template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=PromptTemplate(input_variables=["query", "answer"], template="Question: {query}\nAnswer: {answer}"),
        input_variables=["query"],
        prefix="""The leases chatbot answers questions relating to ASC 842 under US GAAP. Please respond to the queries as shown in 
        the example, with a response followed by reference from multiple sources with either page numbers or section numbers. 
        The responses should be provided only from the provided PDF documents.  The responses should be clear and helpful and should use 
        language that is easy to understand. Responses should include examples and potential scenarios.  If the answer is not avaiable in
        provided PDF documents, the response should be "I do not have information related to that specific scenario, please seek guidance
        from a qualified expert." Additionally, if someone asks a question unrelated to lease accounting, kindly and gently guide them back
        to the topic of lease accounting. """,
        suffix="\n\nQuestion: {query}\nAnswer:",
        example_separator="\n\n"   
    )
    return template

#initialize Conversational Retrieval Chain
def initialize_crc(vector_store, prompt_template):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=openai_api_key)
    retriever = vector_store.as_retriever()
    crc = ConversationalRetrievalChain(llm=llm, retriever=retriever, prompt_template=prompt_template)
    return crc

#Initialize history before it is accessed
if 'history' not in st.session_state:
    st.session_state['history'] = []

# define streamlit app
def main():
    if 'history' not in st.session_state:
        st.session_state['history']=[]
    
    
    examples = create_examples()
    prompt_template = setup_prompt_template(examples)
    
    try:
        # Load and prepare documents
        if 'documents' not in st.session_state:
            text_files = load_files_from_directory('PDFs_and_TXT')
            # pdf_texts = load_pdfs_from_directory('pdfs')
            documents = [text for text in text_files.items()]
            st.session_state.documents = documents
            vector_store = setup_vector_store(documents)
            crc = initialize_crc(vector_store, prompt_template)
            st.session_state.crc = crc
        
        
        
        question = st.text_input("Ask a question about lease accounting:")
        if question and 'crc' in st.session_state:
            crc = st.session_state.crc
            # Display a spinner while processing the question
            with st.spinner("Searching for the answer..."):
                response = crc.run({'query': question, 'chat_history': st.session_state['history']})
                print("appending question and response to history")
                print("Question:", question)
                print("Response:", response)
                st.session_state['history'].append((question, response))
                print("appended successfully")
                st.write(response)
                
        if question:
            #add debugging statements
            print("Chat history:", st.session_state['history'])
            print("Question:", question)
            #add history management
            # if 'history' not in st.session_state or not isinstance(st.session_state['history'], list):
            #     st.session_state['history'] = []
            
        for prompts in st.session_state['history']:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])
        
    except Exception as e:
        #add debugging statement
        print("Error:", e)
        st.error(f"An error occurred: {str(e)}")    
        
if __name__ == "__main__":
    main()


