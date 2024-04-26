
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
import PyPDF2
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain, LLMChain
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

# Create titles and headers seen on app
st.title('ASC 842 AI Assistant')
st.header('This assistant is preloaded with accounting guidance related to ASC 842 Leases under US GAAP.')
st.divider()
st.write('This assistant cannot give specific accounting advice. It is a learning tool and a proof of concept. It is not intended for commercial use. For accounting advice, please consult an appropriate professional.')
st.divider()


# Define the directory where source documents are stored
directory = 'PDFs_and_TXT'

# Define function to load source documents from directory
def load_sources(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            doc = loader.load()
            documents.extend(doc)
    return documents


# Define examples to instruct app how to respond
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

# Create template for examples
example_template = PromptTemplate(input_variables=["query", "answer"], template="Question: {query}\nAnswer: {answer}") 


# Create function to setup prompt template
def setup_prompt_template(examples):
    template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        input_variables=["query"],
        prefix="""You are a leases chatbot. You answer questions relating to ASC 842 under US GAAP. You respond to the queries as shown in 
        the examples. Each response will be followed by reference from multiple sources with section numbers from the source documents. 
        The responses will be provided only from the provided PDF source documents.  The responses will be clear and helpful and will use 
        language that is easy to understand. Responses will include examples and potential scenarios.  If the answer is not avaiable in
        the PDF source documents, the response will be "I do not have information related to that specific scenario, please seek guidance
        from a qualified expert." Additionally, if someone asks a question unrelated to lease accounting, kindly and gently guide them back
        to the topic of lease accounting by saying, "that question is outside the scope of what I can respond to, let's get
        back to lease accounting, how can I help you?" """,
        suffix="\n\nQuestion: {query}\nAnswer:",
        example_separator="\n\n"   
    )
    return template


#initialize conversational retrieval chain
def initialize_crc(embeddings, documents, prompt_template):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents) 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='db')
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=openai_api_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    retriever=vector_store.as_retriever(metadata_fields=['metadata'])
    retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    
    #combined chain combines LLMChain and conversational retrieval chain
    combined_chain = LLMChain([llm_chain, retrieval_chain])
    return combined_chain



# define streamlit app
def main():
    try:
        
        #Initialize history before it is accessed
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        if 'documents' not in st.session_state:
            st.session_state['documents'] = load_sources('PDFs_and_TXT')
        
        if 'chain' not in st.session_state:
            embeddings = OpenAIEmbeddings()
            prompt_template = setup_prompt_template(examples)
            st.session_state['chain'] = initialize_crc(embeddings, st.session_state['documents'], prompt_template)
        
        question = st.text_input('Ask a question about lease accounting:')
        st.divider()

        if question:
            with st.spinner("Searching the guidance..."):
                response = st.session_state['chain'].run({'question':question,'chat_history':st.session_state['history'], 'metadata': True})
                st.session_state['history'].append((question,response))
                st.write(response)
                st.divider()
                    
                st.write('Conversation History')
                for prompts in st.session_state['history']:
                    st.write("Question: " + prompts[0])
                    st.write("Answer: " + prompts[1]) 
    
    except Exception as e:
        #add debugging statement
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
