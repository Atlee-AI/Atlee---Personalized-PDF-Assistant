from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain, LLMChain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.docstore.document import Document

from collections import Counter

from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import InMemoryStore
from langchain_anthropic import ChatAnthropic
from langchain_ai21 import ChatAI21
from langchain_community.chat_models import ChatLiteLLM

from tqdm.autonotebook import tqdm

from langchain.llms import Ollama

from PIL import Image

import pandas as pd

import anthropic
import base64

import uuid
import io

import asyncio

import time
import os

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader



with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    print(config)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

def logout():
    authenticator.logout()


user_personas = {}

user_id = str(uuid.uuid4())
user_personas[user_id] = {}

dir_path = os.getcwd() + '/.env'
load_dotenv(dir_path)

interests = ''


PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
PINECONE_API_KEY=os.environ['PINECONE_API_KEY']
ANTHROPIC_API_KEY=os.environ['ANTHROPIC_API_KEY']
AI21_API_KEY=os.environ["AI21_API_KEY"]
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
# ANTHROPIC_API_KEY='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'
LLAMA_API_KEY='LL-w5KOiBu1f11QJb7xYGk0iQ32LXkpt8pEdJmLrraHEDG6hg4h7E0XB5Kc5TygtEdJ'

EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
CHAT_MODEL='claude-3-opus-20240229'
IMAGE_MEDIA_TYPE='image/jpeg'

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
chat_model = ChatAnthropic(temperature=0.6, api_key=ANTHROPIC_API_KEY, model=CHAT_MODEL, default_headers={"anthropic-beta": "tools-2024-04-04"},)
# chat_model = Ollama(base_url="http://localhost:11434", model="llama3", temperature=0.7)
# chat_model = ChatAI21(model="j2-ultra", temperature = 0.7, max_tokens=8190)
# chat_model = ChatLiteLLM(model="gpt-3.5-turbo", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY)
processed_document = []

def get_one_line_interest_from_history(search_titles, intrest_from_user):
    prompt_text="""
    {search_titles}
    
    These are the recent search history titles of a person.
    
    {intrest_from_user}
    These are the interests that the person has chosen manually when asked.
    
    Give a concise overall interest of the above person in only one line in readable form.
    """
    prompt = PromptTemplate(
        input_variables=[
            "search_titles",
            "intrest_from_user",
        ],
        template=prompt_text,
    )
    interest_chain = LLMChain(
        llm=chat_model,
        prompt=prompt,
        verbose=False
    )
    
    return interest_chain.invoke(
        {
            "search_titles": search_titles,
            "intrest_from_user": intrest_from_user,
        },
        return_only_outputs=True
    )['text']

def get_interests(df, interest_from_user):
    sorted_df = df.sort_values(by='Timestamp', ascending=False)
    if len(sorted_df)>100:
      nrows = len(sorted_df)/20
      top_df = sorted_df.iloc[:nrows]
    elif len(sorted_df)>10:
      top_df = sorted_df.iloc[:10]
    elif len(sorted_df)<10:
      top_df = sorted_df.iloc[:len(sorted_df)]

    interests = get_one_line_interest_from_history(top_df['Title'].tolist(), interest_from_user)
    return interests


class Element(BaseModel):
    type: str
    text: Any

def create_pinecone_index(index_name):
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]
    if index_name in existing_indexes:
        pc.delete_index(index_name)
    pc.create_index(
        index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pc.Index(index_name)
    time.sleep(1)
    index.describe_index_stats()

def get_text_elements(categorized_elements):
    text_elements = [e for e in categorized_elements if e.type == "text"]
    texts = [i.text for i in text_elements]
    return texts

def get_table_elements(categorized_elements):
    table_elements = [e for e in categorized_elements if e.type == "table"]
    tables = [i.text for i in table_elements]
    return tables

def fetch_local_image(image_path):
  try:
    image = Image.open(image_path)

    with io.BytesIO() as output:
      image.save(output, format="JPEG")
      image_data = output.getvalue()

    base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
    return base64_encoded_image

  except FileNotFoundError:
    raise FileNotFoundError(f"Image file not found: {image_path}")

def get_image_summaries_from_anthropic(image_media_type, local_image):
  message = anthropic_client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=1024,
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "source": {
                          "type": "base64",
                          "media_type": image_media_type,
                          "data": local_image,
                      },
                  },
                  {
                      "type": "text",
                      "text": "Describe this image in one line."
                  }
              ],
          }
        ],
      )
  return message

def get_image_summaries():
  image_summaries = []
  image_path = os.getcwd() + '\\figures\\'
  for images in os.listdir(image_path):
    if not os.path.isdir(images):
      try:
        local_image = fetch_local_image(image_path + images)
        image_summary = get_image_summaries_from_anthropic(IMAGE_MEDIA_TYPE, local_image)
        image_summaries.append(image_summary.content[0].text)
      except FileNotFoundError as e:
        print(f"Error: {e}")
  return image_summaries

def get_overall_summary(raw_data):
  prompt_text="""
  You are an assistant tasked with summarizing the give data which may include tables/text. \
  Give a concise summary of the data. Table or Text chunk: {data}
  """
  prompt = ChatPromptTemplate.from_template(prompt_text)
  summary_chain = (
      {"data": RunnablePassthrough()}
      | prompt
      | chat_model
      | StrOutputParser()
  )
  return [summary_chain.invoke(raw_data)]

def store_pdf_in_pinecone(file_name, index_name, embeddings):
    raw_pdf_elements = partition_pdf(
        filename=file_name,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        image_output_dir_path='',
    )

    print("Elements in the PDF: ", end='\n')
    print(Counter(type(element) for element in raw_pdf_elements), end='\n')

    categorized_elements = []
    for element in raw_pdf_elements:
        categorized_elements.append(Element(type="text", text=str(element)))

    id_key = 'doc_id'

    texts = get_text_elements(categorized_elements)
    tables = get_table_elements(categorized_elements)
    img_summaries = get_image_summaries()

    text_summaries = get_overall_summary(texts)
    table_summaries = get_overall_summary(tables)

    text_data = texts + text_summaries
    table_data = tables + table_summaries

    doc_ids = [str(uuid.uuid4()) for _ in text_data]
    table_ids = [str(uuid.uuid4()) for _ in table_data]
    img_ids = [str(uuid.uuid4()) for _ in img_summaries]

    text_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_data)
    ]

    table_docs = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_data)
    ]

    img_summary_docs = [
        Document(page_content=s, metadata={id_key: img_ids[i]})
        for i, s in enumerate(img_ids)
    ]
    
    # processed_document = text_docs + table_docs + img_summary_docs

    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=store,
        id_key=id_key,
    )

    vector_store.add_documents(text_docs)
    retriever.docstore.mset(
        list(
            zip(
                doc_ids, text_data
            )
        )
    )

    vector_store.add_documents(table_docs)
    retriever.docstore.mset(
        list(
            zip(
                table_ids, table_data
            )
        )
    )

    vector_store.add_documents(img_summary_docs)
    retriever.docstore.mset(
        list(
            zip(
                img_ids, img_summaries
            )
        )
    )

    return retriever

def get_personalized_output(question, retriever, chat_model, user_personas, user_id):
    print("Getting personalized output..!", end='\n\n')
    user_persona = user_personas[user_id]
    
    prompt_template="""
    {chat_history}
    
    Answer the question based on the following context (if context is empty refer previous conversation), which may or may not include texts and tables:
    {context}.
    
    Question:
    {question}
    
    Explain to the below user,
    {interests}
    
    Give personalized answer to the above person. Don't mention the interests of the user explicitly in your response.
    """
    
    prompt = PromptTemplate(
        input_variables=[
            "context", 
            "question",
            "interests",
        ],
        template=prompt_template,
    )
    
    memory=ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        k=3,
    )
    
    chain = LLMChain(
        llm=chat_model,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    result = chain.invoke(
        {
            "interests": user_persona["web_interests"],
            "context": retriever,
            "question": question,
        },
        return_only_outputs=True,
    )
    return result['text']


def main():
    global retriever_obj
    authenticator.login()
    create_pinecone_index(PINECONE_INDEX_NAME)

    if st.session_state["authentication_status"]:
        if "name" in st.session_state:
            st.sidebar.header(f'Welcome *{st.session_state["name"]}*')
            st.sidebar.divider()

            with open("interests.txt", "r") as line:
                interests = line.readline().split(',')
            
            interests.sort()
            
            interest_from_user = st.sidebar.multiselect(
                'Choose Interest to Personlize your AI',
                interests,
            )

            print(interest_from_user)
            
            col, _ = st.sidebar.columns(2)

            with col:
                authenticator.logout()
                
            
            uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf", "png", "jpeg"))

            if uploaded_file is not None:
                
                FILE_NAME = "files/"+ st.session_state["username"] + "_"+ uploaded_file.name
                with open(FILE_NAME, "wb") as f:
                    f.write( uploaded_file.read())

                st.success("File saved successfully!")
                
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.retriever = []
                st.session_state.counter = 0
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Let me know what's in your Mind", disabled=not uploaded_file):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

            if uploaded_file and prompt:
                query = prompt
                retriever = retriever_obj or []
                if st.session_state.counter == 0:
                    retriever = store_pdf_in_pinecone(FILE_NAME, PINECONE_INDEX_NAME, embeddings)
                    retriever.search_type = SearchType.mmr
                    retriever_obj = retriever
                time.sleep(5)
                response = retriever.vectorstore.similarity_search_with_score(query)
                df = pd.read_csv('out.csv')
                interests = get_interests(df, interest_from_user)
                user_personas[user_id]["web_interests"] = interests
                print("response", end="\n")
                print(response)
                if response:
                    if response[0][1] < 0.1:
                        response = ''
                    else:
                        response = [response[i][0].page_content for i in range(len(response))]
                else:
                    response = ''
                result = get_personalized_output(query, response, chat_model, user_personas, user_id)
                with st.chat_message("ai"):
                    st.markdown(result)
                st.session_state.messages.append({"role": "ai", "content": result})
                st.session_state.counter += 1
                print(st.session_state)
                            
        else:
            st.sidebar.write("Please log in to view the content.")
            st.sidebar.title("Atlee Personalized AI")

        
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        if st.session_state["authentication_status"]:
            try:
                if authenticator.reset_password(st.session_state["username"]):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    retriever_obj = None
    main()