# First
import shutil
import subprocess
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
# import requests
import tiktoken
import getpass
from streamlit_feedback import streamlit_feedback
from PIL import Image
import base64

from huggingface_hub import InferenceClient


from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI, HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
# import openai
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import load_index_from_storage
from llama_index.storage import StorageContext
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.docstore import BaseDocumentStore

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    GPTVectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
# from transformers import AutoModel, AutoTokenizer
# import time

import time
# print(os.getcwd())
from utils import gaia_api
from utils import gaia_utils

from tempfile import NamedTemporaryFile
import os

from os import listdir
from os.path import isfile, join

import pandas as pd
import GetRemoteClientIP as getip

#URI = "http://127.0.0.1:8080/generate_stream"
#URI = "http://192.168.200.8:8080/generate_stream"
URI = "http://gaia-u-01.westeurope.cloudapp.azure.com:8080"

DataPath = 'data'
_1_HOUR = 3600

# Set the background color using the set_page_config function
st.set_page_config(layout="wide",
                   page_title="Gen AI for Aerospace",
                   page_icon=":brain:",
                   initial_sidebar_state="expanded"
                  )

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
#     model_name="StabilityAI/stablelm-tuned-alpha-3b",
#     device_map="auto",
#     stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )

#@st.cache_data(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    return st.markdown(page_bg_img, unsafe_allow_html=True)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def does_file_exist_in_dir(path):
    return any(isfile(join(path, f)) for f in listdir(path))

def ConvertExcelToTxt(file):
    try:
        read_file = pd.read_excel(file, sheet_name=None)
        for key, val in read_file.items():
            val.to_csv(r'pandas.txt', index=None, header=True, mode='a')
    except Exception as ex:
        print(ex)

    return 'pandas.txt'

def WriteGreeting(prompt):
    pre_prompt = []
    pre_prompt.append({"role": "user", "content": prompt})

    full_prompt = gaia_utils.llama_v2_prompt(pre_prompt)
    print(full_prompt)

    # now fill the message into the chat history above
    #st.chat_message("user").write(prompt)

    #i =+ 1
    #streamlit_feedback(
    #    feedback_type="thumbs",
    #    optional_text_label="[Optional] Please provide an explanation",
    #    key = str(i),
    #    on_submit=gaia_api.provide_feedback
    #)

    client = InferenceClient(URI)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        emojitoken = "\n"
        STATE = 0

        # send to the TGI server
        try:
            for token in client.text_generation(full_prompt, temperature=0.9, max_new_tokens=250, stream=True):
                if STATE == 0:
                    if '*' in token:
                        emojitoken += token
                        STATE = 1
                    else:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                elif STATE == 1:
                    if '*' in token:
                        emojitoken += token + '\n'
                        full_response += emojitoken
                        message_placeholder.markdown(full_response + "▌")
                        STATE = 0
                    else:
                        emojitoken += token
                print(token)
            # add everything besides the last EOS token
            message_placeholder.markdown(full_response[:-4])
        except:
            print('failed to get data from server')

    # add the model reply to the list of messages
    #st.session_state.messages.append({"role": "assistant", "content": full_response})

def SubmitIndex():
    with st.form("my_form"):
        st.success("You index submitted: " + gaia_utils.UserInput, icon="✅")

def ShowPopUpWindows():
    # Create a form where users can input text
    with st.form("my_form", clear_on_submit=False):
        st.write("Please enter your index name:")
        gaia_utils.UserInput = st.text_input("Insert name here")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit", on_click=SubmitIndex)
        if submitted:
            st.write("You index submitted:", gaia_utils.UserInput)
            # You can handle the submitted text in any way you need here
            # For example, you could store it in a database or file

@st.cache_resource(show_spinner=False, experimental_allow_widgets=True)
def load_data_from_file(uploaded_file):
    with st.spinner(text="Loading Embedding"):
        StartEmbedding = time.time()
        print(f"loading embedding... {time.time()}")
        embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        print(f"loaded embedding! {time.time()}")

    qnt = len(uploaded_file)
    for upload_file in uploaded_file:
        mysuffix = '.' + upload_file.name.split('.')[-1]
        with NamedTemporaryFile(dir='.', delete=False, suffix='.' + upload_file.name.split('.')[-1]) as f:
            f.write(upload_file.getbuffer())
            print(upload_file.name + ' is uploaded now')
            if '.xls' in mysuffix:
                print('Go to convert Excel file to text file...')
                myname = ConvertExcelToTxt(f.name)
                newuploadname = upload_file.name.split('.')
                shutil.copy(myname, DataPath + '//' + newuploadname[0] + '.txt')
                os.remove(myname)
            else:
                shutil.copy(f.name, DataPath + '//' + upload_file.name)
            # docs = reader.load_data()
        name = f.name.split('\\')
        os.remove(name[len(name) - 1])

    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()

    # with st.spinner(text="Loading index"):
    #     print(f"reading file {time.time()}")
    #     if upload_file:
    #         # load the data from memory without storing it in disk
    #         mysuffix = '.' + upload_file.name.split('.')[-1]
    #         with NamedTemporaryFile(dir='.',delete=False, suffix='.' + upload_file.name.split('.')[-1]) as f:
    #             f.write(upload_file.getbuffer())
    #             reader = SimpleDirectoryReader(input_files=[f.name])
    #             print(f.name)
    #             docs = reader.load_data()
    #         os.remove(f.name)
    #     else:
    #         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    #         docs = reader.load_data()

    print(f"loading index {time.time()}")
    #service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)

    #RemoveIndex()
    # # delete previouse index and nodes
    # if gaia_utils.INDEX != '':
    #     gaia_utils.INDEX.delete_ref_doc(ref_doc_id=gaia_utils.DOC_ID)
    #     gaia_utils.NODES_ID = []
    #     if "retriever_engine" in st.session_state.keys():
    #         st.session_state.pop('retriever_engine')
    #if len(st.session_state.messages) > 0:
    #    st.session_state.messages.clear()

    st.session_state.messages.clear()
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    #ShowPopUpWindows()
    index.storage_context.persist(persist_dir='VDB\\MSDS-Docs')
    gaia_utils.DOC_ID.clear()
    gaia_utils.NODES_ID.clear()
    for doc in docs:
        gaia_utils.DOC_ID.append(doc.doc_id)
    gaia_utils.INDEX_ID = index.index_id
    for key, value in index.docstore.docs.items():
        gaia_utils.NODES_ID.append(key)
    gaia_utils.INDEX = index
    # index.storage_context.persist("./" + upload_file.name )
    TotalTime = round(time.time() - StartEmbedding, 1)
    print(f"index created in {TotalTime} sec.")
    if "retriever_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.retriever_engine = index.as_retriever(verbose=True)

    #st.cache_resource.clear()

    return index

@st.cache_resource(show_spinner=False)
def load_persistent_data(persist_dir):
    Start = time.time()
    st.session_state.messages.clear()
    index_cache_web_dir = Path(persist_dir)
    if not index_cache_web_dir.is_dir():
        index_cache_web_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner(text="Loading the knowledge base. – hang tight! This should take 1-2 minutes."):
        print('Start loading...')
        Start = time.time()
        embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)
        storage_context = StorageContext.from_defaults(persist_dir=str(index_cache_web_dir))
        index = load_index_from_storage(
            index_store=SimpleIndexStore.from_persist_dir(persist_dir='./' + persist_dir),
            storage_context=storage_context,
            service_context=service_context
        )
        gaia_utils.DOC_ID.clear()
        gaia_utils.NODES_ID.clear()
        for key, val in index.ref_doc_info.items():
            gaia_utils.DOC_ID.append(key)
        gaia_utils.INDEX_ID = index.index_id
        for key, value in index.docstore.docs.items():
            gaia_utils.NODES_ID.append(key)
        gaia_utils.INDEX = index
        TotalTimeLoading = round(time.time() - Start, 1)
        print('loaded from storage - ' + str(TotalTimeLoading) + ' sec')
        if "retriever_engine" not in st.session_state.keys():  # Initialize the chat engine
            try:
                st.session_state.retriever_engine = index.as_retriever(verbose=True)
            except Exception as ex:
                print(ex)

def RemoveIndex():
    if gaia_utils.INDEX != '':
        for file_upload in uploaded_file:
            file_upload.close()
        uploaded_file.clear()
        #st.experimental_rerun()
        for doc in gaia_utils.DOC_ID:
            gaia_utils.INDEX.delete_ref_doc(ref_doc_id=doc)
        gaia_utils.NODES_ID = []
        gaia_utils.INDEX = ''
        st.session_state[str(st.session_state["file_uploader_key"])] = []
        #st.session_state.pop(str(st.session_state["file_uploader_key"]))
        st.session_state["file_uploader_key"] += 1
        if "retriever_engine" in st.session_state.keys():
            st.session_state.pop('retriever_engine')
        if len(st.session_state.messages) > 0:
            st.session_state.messages.clear()
        for file in os.listdir(DataPath):
            os.remove(DataPath + '//' + file)
        print('delete previouse index and nodes')

def initialize_session_state():
    if "no_file_conversation" not in st.session_state:
        pass
        #st.session_state.no_file_conversation = [
        #    SystemMessage(content="I'm a PDF Chatbot. Ask me a question about your documents!")
        #]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'user_input_prompt' not in st.session_state: # This one is specifically use for clearing the user text input after they hit enter
        st.session_state.user_input_prompt = 'None'
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = None
    if "is_vectorstore" not in st.session_state:
        st.session_state.is_vectorstore = False
    if "extra_class" not in st.session_state: # This is a control variable, use to check the type of user's last conversation, either 'Warning' or 'None'
        st.session_state.extra_class = None

def click_button():
    #RemoveIndex()
    gaia_api.prev_topics.clear()
    gaia_utils.MyModelHistory.clear()
    if gaia_utils.INDEX != '':
        st.cache_resource.clear()
        st.session_state.clear()

        for doc in gaia_utils.DOC_ID:
            gaia_utils.INDEX.delete_ref_doc(ref_doc_id=doc)
        gaia_utils.DOC_ID.clear()
        gaia_utils.NODES_ID.clear()
        gaia_utils.INDEX = ''

        if gaia_utils.FILE_UPLOAD_KEY == 0: gaia_utils.FILE_UPLOAD_KEY = 2
        else: gaia_utils.FILE_UPLOAD_KEY += 1
        st.session_state.file_uploader_key = gaia_utils.FILE_UPLOAD_KEY

        for file in os.listdir(DataPath):
            os.remove(DataPath + '//' + file)
        print('delete previouse index and nodes')
    print('Pressed button...')

def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    return user_response.update({"some metadata": 123})

# design section - sidebar definition and selections
# Here is start!
#######################################
with st.sidebar:
    ip = getip.get_remote_ip()
    username = getip.GetUserName(ip)

    if "file_uploader_key" not in st.session_state.keys():
        st.session_state["file_uploader_key"] = gaia_utils.FILE_UPLOAD_KEY
    if "uploaded_files" not in st.session_state.keys():
        st.session_state["uploaded_files"] = []

    image_url = 'GAIA_black.png'
    image = Image.open(image_url)
    st.image(image)

    original_title = '<p style="font-family:prometo trial; color:white; font-size: 20px; font-weight: bold; ">Model Settings</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    #st.header(':gray[_Model Settings_]', divider='green')
    model_opts = gaia_api.get_models()
    vector_opts = gaia_api.get_vectors()
    model_sel = st.selectbox(label=":blue[Select a model]",options=model_opts['names'])

    if gaia_utils.MyModel != model_sel:
        gaia_utils.MyModelHistory[gaia_utils.MyModel] = gaia_api.prev_topics
        gaia_api.prev_topics = gaia_utils.MyModelHistory.get(model_sel, [])
        if "messages" in st.session_state:
            gaia_utils.MyModelHistoryMsg[gaia_utils.MyModel] = st.session_state.messages
            st.session_state.messages = gaia_utils.MyModelHistoryMsg.get(model_sel, [])
        gaia_utils.MyModel = model_sel

    model_URI = model_opts['URI'][model_opts['names'].index(model_sel)]
    with st.expander(label=":blue[Additional Knowledge]", expanded=False):
        vector_sel = st.selectbox(":blue[Knowledge base]", vector_opts)
        if gaia_utils.VectorSelected != vector_sel and vector_sel != 'None':
            if does_file_exist_in_dir('VDB\\' + vector_sel):
                load_persistent_data('VDB\\' + vector_sel)
        gaia_utils.VectorSelected = vector_sel

        uploaded_file = st.file_uploader(
            ":blue[Upload a document]",
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
            type=("txt", "md", "docx", "pdf", "xls", "xlsx")
        )

    st.session_state["uploaded_files"] = uploaded_file

    #if st.session_state["file_uploader_key"] == 1:
    #    st.session_state["file_uploader_key"] = 0

    with st.expander(label=":blue[Custom Instructions]", expanded=False):
        personal_inst = st.text_area("What would you like GAIA to know about you to provide better responses?" , max_chars=2000)
        chracter_inst = st.text_area("How would you like GAIA to respond?", max_chars=2000)
    
    #username = getpass.getuser()
    ip = getip.get_remote_ip()
    username = getip.GetUserName(ip)

    creativity_inst = st.select_slider(':blue[Creativity]', options=['Precise', 'Balanced', 'Creative'])
    prev_topics = gaia_api.prev_topics.get(username, [])
    if len(prev_topics) > 0:
        original_title = '<p style="font-family:prometo trial; color:white; font-size: 20px; font-weight: bold; ">Previous Chats</p>'
        #st.markdown(original_title, unsafe_allow_html=True)
        st.header(f':gray[_Previous Chats_] of {username}', divider='green')
        #prev_topics = gaia_api.get_chat_titles()
        st.selectbox(':blue[Previous Chats]', prev_topics)
    label = ":black[Start New Chat]"    # colors: blue, green, orange, red, violet, gray/grey, rainbow.
    new_chat_press = st.button(label=label, type="primary", on_click=click_button, key='new_chat',use_container_width=True)
    
    # Add CSS styles to control the width of the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    gaia_utils.ST = st
# design section - main screen elements
#######################################
#
set_png_as_page_bg('Gaia_Watermark.png')
# Image   
#image_url = 'GAIA_AI_Image1_240x160.jpg'
#image_url = 'GAIA_logo.png'
#image = Image.open(image_url)
#st.image(image)
# Set background color

original_title = '<p style="font-family:prometo trial; color:#0790e6; font-size: 20px; font-weight: bold; ">Hello, my name is GAIA</p>'
st.markdown(original_title, unsafe_allow_html=True)

# title message
if model_sel == "AURA":
    #st.title("I am your project's expert - ask me anything...")
    title = "I am your project's expert - ask me anything..."
    original_title = '<p style="font-family:prometo trial; color:Black; font-size: 40px; font-weight: bold; ">I am your project expert - ask me anything...</p>'
    st.markdown(original_title, unsafe_allow_html=True)
elif model_sel == "GAIA":
    #st.title("GAIA - Let's build the aerospace future together...")
    title = "GAIA - Let's build the aerospace future together..."
    original_title = '<p style="font-family:prometo trial; color:Black; font-size: 40px; font-weight: bold; ">GAIA - Lets build the aerospace future together...</p>'
    st.markdown(original_title, unsafe_allow_html=True)
elif model_sel == "Llama-2":
    #st.title("Ask me anything you want")
    title = "Ask me anything you want"
    original_title = '<p style="font-family:prometo trial; color:Black; font-size:40px; font-weight:bold; ">Ask me anything you want</p>'
    st.markdown(original_title, unsafe_allow_html=True)
elif model_sel == "WizardCoder":
    #st.title("Let's write great code together...")
    title = "Let's write great code together..."
    original_title = '<p style="font-family:prometo trial; color:Black; font-size: 40px; font-weight: bold; ">Lets write great code together...</p>'
    st.markdown(original_title, unsafe_allow_html=True)
# chat prompt

prompt = st.chat_input(placeholder="Enter your message here...", max_chars=2000)

if prompt:
    gaia_api.prev_topics.append(prompt)

#background_color = "<style>body {background-color: #303a2f;}</style>"
#st.markdown(background_color, unsafe_allow_html=True)

#new_topic = st.button('New Topic', key='new_topic',use_container_width=False)

# app logic section
#######################################
if uploaded_file:
    index = load_data_from_file(uploaded_file)

if vector_sel:
    pass

# initialize the messages list
if "messages" not in st.session_state:
    st.session_state.messages = []
    #st.chat_message("assistant").write("How can I help you today?")

# fill the messages list in the screen
#for msg in st.session_state.messages:
#    st.chat_message(msg["role"]).write(msg["content"])
if len(st.session_state.messages) > 0:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        #st.chat_message(st.session_state.messages[len(st.session_state.messages) - 1]["role"]).write(st.session_state.messages[len(st.session_state.messages) - 1]["content"])
i = 0

myflag = gaia_utils.OnesGreetingFlag.get(username, time.time() - (24 * _1_HOUR))
if (time.time() - myflag) >= (12 * _1_HOUR):
    for file in os.listdir(DataPath):
        os.remove(DataPath + '//' + file)
    for file in os.listdir('VDB'):
        for doc in os.listdir('VDB' + '//' + file):
            os.remove('VDB' + '//' + file + '//' + doc)


    gaia_utils.OnesGreetingFlag[username] = time.time()
    WriteGreeting("Write a greeting of a few words and add a daily joke that an engineer would appreciate")
    #ShowPopUpWindows()

if prompt:
    # add all previous messages to the prompt
    pre_prompt = st.session_state.messages

    # verify we are using the correct model now
    if model_URI:
        URI = model_URI
        print(model_URI)
    
    # URI = "http://127.0.0.1:8080/generate_stream"

    # prepare the template we will use when prompting the AI
    template = """Please use the following information to answer the user’s question.
    If you don’t know the answer, please say that you don’t know. Do not make up an answer.
    Context: {context}
    Question: {prompt}
    Helpful answer:
    """

    # if we need - add context to the prompt using the retriever engine
    if uploaded_file and ("retriever_engine" in st.session_state.keys()):
        nodes = st.session_state.retriever_engine.retrieve(prompt)
        context = "context: "
        # TODO: use the best nodes to get the context
        context = context + nodes[0].text
        context_prompt = 'Use the following pieces of information to answer the user\'s question.\n' + \
                         'If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer.\n' + \
                         'Context: ' + context + '\n' + \
                         'Question: ' + prompt + '\n' + \
                         'Only return the helpful answer below and nothing else.\n' + \
                         'Helpful answer:'
        #context_prompt = "Answer my question using the provided context.\n" + context + "\nmy question: " + prompt
        pre_prompt.append({"role": "user", "content": context_prompt})
    elif "retriever_engine" in st.session_state.keys():
            nodes = st.session_state.retriever_engine.retrieve(prompt)
            context = "context: "
            # TODO: use the best nodes to get the context
            context = context + nodes[0].text
            context_prompt = 'Use the following pieces of information to answer the user\'s question.\n' + \
                             'If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer.\n' + \
                             'Context: ' + context + '\n' + \
                             'Question: ' + prompt + '\n' + \
                             'Only return the helpful answer below and nothing else.\n' + \
                             'Helpful answer:'
            #context_prompt = "Answer my question using the provided context.\n" + context + "\nmy question: " + prompt
            pre_prompt.append({"role": "user", "content": context_prompt})
    else:
        pre_prompt.append({"role": "user", "content": prompt})

    # TODO: use tiktoken to clip the messages until the number of tokens is less than 2000
    # check the size of each messages from the newest to the oldest
    sum_tokens = 0
    for msg in reversed(pre_prompt):
        if sum_tokens <= 1200:
            num_tokens = gaia_utils.num_tokens_from_string(msg["content"], "cl100k_base")
            sum_tokens+=num_tokens
            print('num token: ' + str(num_tokens) + ' | sum token: ' + str(sum_tokens))
        if sum_tokens > 1200:
            # remove the message from the dict 
            pre_prompt.remove(msg)

    full_prompt = gaia_utils.llama_v2_prompt(pre_prompt)
    print(full_prompt)

    # now fill the message into the chat history above
    st.chat_message("user").write(prompt)

    client = InferenceClient(URI)

    DataIsCorrectly = True
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # send to the TGI server
        try:
            for token in client.text_generation(full_prompt, temperature=0.5, max_new_tokens=1024, stream=True):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
                # print(token)
            # add everything besides the last EOS token
            message_placeholder.markdown(full_response[:-4])
        except:
            DataIsCorrectly = False
            print('failed to get data from server')

    if DataIsCorrectly:
        #i += 1
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            #key=str(i),
            key='1',
            on_submit=_submit_feedback
            # on_submit=gaia_api.provide_feedback
        )
        if feedback:
            print('feedback')

    # add the model reply to the list of messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})