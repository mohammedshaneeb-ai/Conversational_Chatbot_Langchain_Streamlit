from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
PineCone_KEY = os.environ.get('PineCone_KEY')
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(
    api_key = PineCone_KEY,
    environment = 'gcp-starter'
)

index = pinecone.Index('langchain-chatbot-streamlit')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = client.completions.create(model="text-davinci-003",
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base. \n\nConversation LOG: \n{conversation}\n\nQuery:{query}\n\n Refined Query: ",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
