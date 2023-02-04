import pandas as pd
import numpy as np
import openai
import pinecone
import os
import streamlit as st

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

st.write('''If you could ask Krishna a question, what would it be?''')
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
    return st.markdown(
    # <div class="container-fluid">
    #     <div class="row align-items-start">
    #          <div  class="col-md-12 col-sm-12">
    #              <span style="color: #808080;">
                     # <small>
    context
        # </small>
     #             </span>
     #         </div>
     #    </div>
     # </div>
     #    , unsafe_allow_html=True)
    )

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": 'text-davinci-003',
}

header = """You are Krishna from Mahabharata, and you're here to answer any question or dilemma of anyone.
    Analyze the person's question to identify the base emotion and root for this emotion, and 
     with this emotion you observed, identify which verses below apply best to their situation and empathetically
     answer their question by summarizing how these verses apply to their situation. \n\nContext:\n"""

def print_verse(q):
    k=[]
    embed = get_embedding(q)
    for i in range(5):
        k.append(int(st.session_state_index.query(embed, top_k=5)['matches'][i]['id']))
    return k    

def return_all_verses(x):
    versee = []
    for i in verse_numbers:
        versee.append(f"{df_index['index'][i]} \n")
    return versee
        
st.write('''Some examples:
1. I've worked very hard but I'm still not able to achieve the success I hoped to, what do I do?
2. I made a million dollars manipulating the stock market and I'm feeling great.
3. How can I attain a peace of mind?
''')



question=st.text_input('How are you feeling?','')
if question!='':
    st.write('Bhagvad Gita says: ') 
    verse_numbers = print_verse(question)
    verses = return_all_verses(verse_numbers)
    verse_strings = "".join(return_all_verses(question))
    prompt = f'''{header}{question}{verse_strings}'''

    response = openai.Completion.create(
        prompt = prompt,
        **COMPLETIONS_API_PARAMS
    )

    st.markdown(response["choices"][0]["text"].strip(" \n"))
    st.markdown('\n\n')
    st.markdown("Relevant verses:")
    st.markdown(verse_strings.replace('\n','\n\n'))

st.write('''Note: This is an AI model trained on Bhagvad Gita and it generates responses from that perspective''')
