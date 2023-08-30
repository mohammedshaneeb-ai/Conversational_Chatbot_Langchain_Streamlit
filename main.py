from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate,MessagesPlaceholder

import streamlit as st
from streamlit_chat import message
from utils  import *
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

st .title("PDF Chatbot")

if 'responses' not in st.session_state:
    st.session_state['response'] = ['How can i assit you']
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    openai_api_key=openai_api_key
    )

if 'buffer_memory' not in st.session_state:
    st.session_state.buffur_memory = ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below ,say 'i don't know' " 
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_template([system_msg_template,MessagesPlaceholder(variable_name='history'), human_msg_template])


conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

 
response_container = st.container()
text_container = st.container()


with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refinerd Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n {query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True,key=str(i)+'_user')


