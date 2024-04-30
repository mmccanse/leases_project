
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAI
from qdrant_client import QdrantClient, models
import qdrant_client
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# Define function to access Qdrant vector store
def get_vector_store():
    #Create a client to connect to Qdrant server
    client = qdrant_client.QdrantClient(
        st.secrets["QDRANT_HOST"],
        api_key=st.secrets["QDRANT_API_KEY"]
        )
    
    #initialize embeddings for vector store
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-3-large"
    )
    
    #create a vector store with Qdrant and embeddings
    vector_store = Qdrant(
        client = client,
        collection_name = st.secrets["QDRANT_COLLECTION_NAME"],
        embeddings = embeddings,
    )
    
    return vector_store

# Define context
context = """You are a leases chatbot. You answer questions relating to ASC 842 under US GAAP. You respond to the queries as shown in 
    the examples. The responses do not have to be brief. Giving a thorough response with citations from the source documents is more
    important than brevity. Each response will be followed by references from multiple sources with section numbers and page numbers (which
    is in the meta data) from the context documents. The response will also include a separate reference to the relevant ASC 842 guidance chapter.
    The responses will be provided only from the provided PDF source context documents. The responses will be clear and helpful and will 
    use language that is easy to understand. Responses will include examples and potential scenarios.  If the answer is not avaiable in the 
    PDF source documents, the response will be "I do not have information related to that specific scenario, please seek guidance from a 
    qualified expert." If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you 
    with. Let's get back to lease accounting."""

# Create function to setup prompt template
def setup_prompt_template():
    prefix="""You are a leases chatbot. You answer questions relating to ASC 842 under US GAAP. You respond to the queries as shown in 
    the examples. The responses do not have to be brief. Giving a thorough response with citations from the source documents is more
    important than brevity. Each response will be followed by references from multiple sources with section numbers and page numbers (which
    is in the meta data) from the context documents. The response will also include a separate reference to the relevant ASC 842 guidance chapter.
    The responses will be provided only from the provided PDF source context documents. The responses will be clear and helpful and will 
    use language that is easy to understand. Responses will include examples and potential scenarios.  If the answer is not avaiable in the 
    PDF source documents, the response will be "I do not have information related to that specific scenario, please seek guidance from a 
    qualified expert." If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you 
    with. Let's get back to lease accounting." 
    
    You will answer the input question based on the provided context:
    
    <context>
    {context}
    </context>
    
    You will use the provided examples for guidance on how to construct your responses. Your responses should be similar."""
     
     # Define examples to instruct app how to respond
    examples = [
        {
            "input": "How do I determine the different lease components?",
            "answer": """Lease components are elements of the arrangement that provide the customer with the right to use an 
            identified asset. An entity should consider the right to use an underlying asset to be a separate lease component 
            if the following 2 conditions are met: 1. Lessee can benefit from the ROU asset either on its own or together with 
            other readily available resources and 2. The ROU is neither highly dependent on; nor highly interrelated with the 
            other ROU(s) in the contract. 
            References: 
            KPMG Leases Handbook, section 4.1, page 151
            PWC Leases Guide, section 2.4, page 2-28
            EY Financial Reporting Developments Lease Accounting, section 1.4, page 27
            ASC: 842-10-15-28 to 15-37, 842-10-15-38 to 15-42C"""
        },
        {
            "input": "I am a lessor, how do I account for lease modifications?",
            "answer": """Several questions must be answered to determine how to appropriately account for lease modifications for 
            a lessor. Is the modified contract a lease, or does it contain a lease? If yes, does the modification result in a 
            separate contract? If yes, account for two separate contracts: the unmodified original contract, and a separate 
            contract accounted for in the same manner as any other new lease. If the modification does not result in a separate 
            contract, remeasure and reallocate the remaining consideration in the contract, reassess the lease classification at 
            the modification effective date, and account for any initial direct costs, lease incentives, and other paymetns made to 
            or by the lessor. Whether or not the lease classification changes, and how it changes drives the appropriate accounting. 
            References: 
            EY - Financial Reporting Developments: Lease Accounting, section 5.6, page 281
            PWC - Leases Guide, section 5.6, page 5-45 
            KPMG - Leases Handbook, section 7.6, page 706
            ASC: 842-10-25-8 to 25-18, 842-10-35-3 to 35-3 to 35-6, 842-10-55-190 to 55-209"""
        }
    ]
    
    #Define format for examples:
    example_format = "\nQuestin: {input}\n\nAnswer: {answer}"
    
    example_prompts = [example_format.format(**ex) for ex in examples]
    
    example_template = PromptTemplate(input_variables=['input', 'context'],
                                      template=example_format)
    
    full_prompt = f"{prefix}\n\n" + "\n\n".join(example_prompts) + "\n\nQuestion: {input}\n\nAnswer: "
    
    # enriched_history = history + [(input, full_prompt)]
    
    #Define suffix for query
    suffix="\n\nQuestion: {input}\nAnswer: "
    
    #Construct FewShotPromptTemplate
    prompt_template = FewShotPromptTemplate(
                                            examples=examples,
                                            example_prompt=example_template,
                                            input_variables=['input','context'],
                                            prefix=prefix,
                                            suffix=suffix,
                                            example_separator="\n\n")
    return prompt_template
    
    
def history_aware_chain(prompt_template,vector_store):    
    # Create new llm instance
    llm = ChatOpenAI(api_key=openai_api_key, model=OPENAI_MODEL, temperature=.1)
    # Set vector_store as retriever
    retriever = vector_store.as_retriever()
    # create history aware retriever that will retrieve relevant 
    # segments from source docs
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, 
        retriever,
        prompt_template)
    return history_aware_chain
    

# create document chain with create_stuff_documents_chain. This 
# tekes the relevant source segments from the history_aware_retriever
# and "stuffs" them into (something?) that the retrieval chain will reference.

def document_chain(prompt_template):
    # Create new llm instance
    llm = ChatOpenAI(api_key=openai_api_key, model=OPENAI_MODEL, temperature=.1)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return document_chain

def retrieval_chain(history_aware_chain, document_chain):
    retrieval_chain = create_retrieval_chain(history_aware_chain, document_chain)
    return retrieval_chain


def process_question(question,crc,history):
    try:
        print("processing question: ", question)
        print("processing chat history: ", history)
        response = crc.run({'question':question,'chat_history':st.session_state['history']})
        if response:
            # If response is True, return a success message
            return response
        else:
        # If response is False, return an error message
            return "Error in response"
    except Exception as e:
        print("Error during processing:", e)
        return str(e), "Error"
        
        
def display_final_response_and_history(final_response, history):
    st.write(final_response)
    st.divider()
    st.markdown(f"**Conversation History**")
    for prompts in reversed(history):
        st.markdown(f"**Question:** {prompts[0]}")
        st.markdown(f"**Answer** {prompts[1]}")
        st.divider
        
#this part will go in main function


#invoke the retrieval chain, this will go in main function
response = retrieval_chain.invoke({'input': input})


# define streamlit app
def main():
    st.title('ASC 842 AI Assistant')
    st.header('This assistant is preloaded with accounting guidance related to ASC 842 Leases under US GAAP.')
    st.divider()
    st.write('This assistant cannot give specific accounting advice. It is a learning tool and a proof of concept. It is not intended for commercial use. For accounting advice, please consult an appropriate professional.')
    st.divider()
    try:
            
        #Initialize history before it is accessed
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        #Initialize vector store
        if 'vector_store' not in st.session_state:
            st.session_state['vector_store'] = get_vector_store()
            st.success('vector store loaded!')
        
        #Initialize prompt template
        if 'prompt_template' not in st.session_state:
            st.session_state['prompt_template'] = setup_prompt_template()
        
        #bring context into session state
        if 'context' not in st.session_state:
            st.session_state['context'] = context
        
        input = st.text_input("""Ask questions about lease accounting. The app will remember your conversation
                              history until you click 'clear history'""", placeholder='Type your question here...')
        st.caption("Press Enter to submit your question.")
        st.divider()

        if input:
            with st.spinner("Searching the guidance..."):
                if 'history_aware' not in st.session_state:
                    st.session_state['history_aware'] = history_aware_chain(st.session_state['prompt_template'],st.session_state['vector_store'])
                if 'document_chain' not in st.session_state:
                    st.session_state['document_chain'] = document_chain(st.session_state['prompt_template'])
                history_aware_retriever_chain.invoke({
                    "chat_history": st.session_state['history'], 
                    "context": st.session_state['context'],
                    "input": input})
                
                response = retrieval_chain.invoke({'input': input})
                st.write(response)
            
            st.divider()
            st.markdown(f"**Conversation History**")
            for prompts in reversed(st.session_state['history']):
                st.markdown(f"**Question:** {prompts[0]}")
                st.markdown(f"**Answer** {prompts[1]}")
                st.divider()
        
    except Exception as e:
        #add debugging statement
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
