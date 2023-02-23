import pandas as pd
import numpy as np
import openai
import pinecone
import streamlit as st
import time
pip install langchain
from langchain.llms import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone.init(
    api_key=pinecone_api_key, 
              environment='us-east1-gcp')

index_name = 'bhagvad-gita'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )
st.session_state_index = pinecone.Index(index_name)

openai.api_key=st.secrets['openai_api_key']


df_index=pd.read_csv('only_verses.csv')

st.write("""
# GitaGPT
""")

st.write('''If you could ask Bhagavad Gita a question, what would it be?''')
st.markdown('\n')
st.markdown('\n')
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def card(context):
    return st.markdown(context)

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 300,
    "model": 'text-davinci-003',
}

header = """You are Krishna from Mahabharata, and you're here to selflessly help and answer any question or dilemma of anyone who comes to you.
    Analyze the person's question below and identify the base emotion and the root for this emotion, and then frame your answer by summarizing how your verses below
    apply to their situation and be emphatetic in your answer."""



def print_verse(q,retries=6):
    k=[]
    embed = get_embedding(q)
    for j in range(retries):
            try:
                for i in range(5):
                    k.append(int(st.session_state_index.query(embed, top_k=5)['matches'][i]['id']))
                return k    
            except Exception as e:
                if j == retries - 1:
                    raise e(st.markdown('Maximum number of retries exceeded'))
                else:
                    st.markdown("Failed to generate, trying again.")
                    time.sleep(2 ** j)
                    continue

def return_all_verses(retries=6):
    versee = []
    for j in range(retries):
        try:
            for i in verse_numbers:
                versee.append(f"{df_index['index'][i]} \n")
            return versee
        except Exception as e:
            if j == retries - 1:
                raise e(st.markdown('Maximum number of retries exceeded'))
            else:
                st.markdown("Failed to generate, trying again.")
                time.sleep(2 ** j)
                continue
               

question=st.text_input("**How are you feeling? Ask a question or describe your situation below, and then press Enter.**",'',placeholder='Type your question here')
# if st.button('Enter'):
if question != '':
    output = st.empty()
    st.write('Bhagvad Gita says: ') 
    verse_numbers = print_verse(question)
    verses = return_all_verses()
    verse_strings = "".join(return_all_verses())
    prompt = f'''{header}\nQuestion:{question}\nVerses:\n{verse_strings}\nAnswer:\n'''

    # response = openai.Completion.create(
    #     prompt = prompt,
    #     **COMPLETIONS_API_PARAMS
    # )
    llm = OpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
    resp = llm(prompt)
    # st.markdown(resp)
    output.markdown(resp)
    # st.markdown(response["choices"][0]["text"].strip(" \n"))
    st.markdown('\n\n')
    st.markdown("Relevant verses:")
    st.markdown(verse_strings.replace('\n','\n\n'))


st.write('''\n\n Here's some examples of what you can ask:
1. I've worked very hard but I'm still not able to achieve the results I hoped for, what do I do?
2. I made a million dollars manipulating the stock market and I'm feeling great.
3. How can I attain a peace of mind?
''')

st.write('\n\n\n\n\n\n\n')

st.write('''Note: This is an AI model trained on Bhagvad Gita and it generates responses from that perspective.''')
