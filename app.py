import chainlit as cl
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from config import aws_region_name, aws_access_key, aws_secret_key
import re


def create_client():
    '''
    Create AWS Bedrock client to interact with LLM
    '''
    bedrock = boto3.client(service_name='bedrock-runtime',
                        region_name=aws_region_name,
                        aws_access_key_id =aws_access_key,
                        aws_secret_access_key =aws_secret_key
                        )
    return bedrock

def create_llm(bedrock_client):
    '''
    Create the LLM, which will be Llama 2 13B Chat from AWS
    '''

    llm = Bedrock(model_id='meta.llama2-13b-chat-v1', 
                  client=bedrock_client,
                  streaming=True,
                  model_kwargs={'temperature':0})
    return llm

def load_pdfs(chunk_size=3000, chunk_overlap=100):
    '''
    Load PDF documents
    '''
    
    loader=PyPDFDirectoryLoader("PDF Documents")
    documents=loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs

def create_vector_store(docs):
    '''
    Build a vector store, using the Titan embedding model from AWS
    '''
    # Set up bedrock client
    bedrock = create_client()
    bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)

    # create and save the vector store
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")
    
    return None

@cl.on_chat_start
async def create_qa_chain():

    # create client 
    bedrock_client = create_client()

    # load llm
    llm = create_llm(bedrock_client=bedrock_client)

    # load embeddings and vector store
    bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock_client)
    vector_store = FAISS.load_local('faiss_index', bedrock_embeddings, allow_dangerous_deserialization=True)
    
    # create memory history
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # create qa chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm, 
                                           chain_type='stuff', 
                                           retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k":3}),
                                           return_source_documents=True,
                                           memory=memory
                                           )
    
    # add custom messages to the user interface
    msg = cl.Message(content="Loading the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the QA Chatbot! Please ask your question."
    await msg.update()
    
    cl.user_session.set('qa_chain' ,qa_chain)

@cl.on_message
async def generate_response(query):
    qa_chain = cl.user_session.get('qa_chain')

    res = await qa_chain.acall(query.content, callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, 
        )])

    # extract results and source documents
    result, source_documents = res['answer'], res['source_documents']

    # Extract all values associated with the 'metadata' key
    source_documents = str(source_documents)
    metadata_values = re.findall(r"metadata={'source': '([^']*)', 'page': (\d+)}", source_documents)

    # Convert metadata_values into a single string
    pattern = r'PDF Documents|\\'
    metadata_string = "\n".join([f"Source: {re.sub(pattern, '', source)}, page: {page}" for source, page in metadata_values])

    # add metadata (i.e., sources) to the results
    result += f'\n\n{metadata_string}'

    # send the generated response to the user
    await cl.Message(content=result).send()