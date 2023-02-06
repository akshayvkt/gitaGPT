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

st.write('''If you could ask Krishna or Arjuna a question, what would it be?''')
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

header = """You are Krishna from Mahabharata, and you're here to selflessly help and answer any question or dilemma of anyone who comes to you.
    Analyze the person's question below and identify the base emotion and the root for this emotion, and then frame your answer by summarizing how the verses below
    apply to their situation and be emphatetic in your answer."""

def print_verse(q):
    k=[]
    embed = get_embedding(q)
    for i in range(5):
        k.append(int(st.session_state_index.query(embed, top_k=5)['matches'][i]['id']))
    return k    

def return_all_verses():
    versee = []
    for i in verse_numbers:
        versee.append(f"{df_index['index'][i]} \n")
    return versee
        

question=st.text_area('**How are you feeling? Ask a question or describe your situation below and press Enter**','',height = 20,placeholder='Type your question here')
if st.button('Enter'):
    st.write('Bhagvad Gita says: ') 
    verse_numbers = print_verse(question)
    verses = return_all_verses()
    verse_strings = "".join(return_all_verses())
    prompt = f'''{header}\nQuestion:{question}\nVerses:\n{verse_strings}\nAnswer:\n'''

    response = openai.Completion.create(
        prompt = prompt,
        **COMPLETIONS_API_PARAMS
    )

    st.markdown(response["choices"][0]["text"].strip(" \n"))
    st.markdown('\n\n')
    st.markdown("Relevant verses:")
    st.markdown(verse_strings.replace('\n','\n\n'))


st.write('''Here's some examples of what you can ask:
1. I've worked very hard but I'm still not able to achieve the results I hoped for, what do I do?
2. I made a million dollars manipulating the stock market and I'm feeling great.
3. How can I attain a peace of mind?
''')

st.write('\n\n\n\n\n\n\n')

st.write('''Note: This is an AI model trained on Bhagvad Gita and it generates responses from that perspective.''')
