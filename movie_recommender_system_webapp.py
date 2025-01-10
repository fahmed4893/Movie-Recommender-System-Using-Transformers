import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pickle
import streamlit as st

df= pd.read_csv('movies.csv')
df.drop('movieId',axis=1,inplace=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
list_embedding= pickle.load(open("C:/Users/Premier UK/movies_list_embeddings_.pkl","rb"))

#query='suggest any romantic movies'
def recommendations(query, list_embedding, df, top=10):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, list_embedding)
    
    top = similarity[0].argsort()[-top:][::-1]
    suggestions= df.iloc[top][['title', 'genres']]
    return suggestions
    

#suggestions = recommendations(query, list_embedding, df)
#print(suggestions[['title', 'genres']])


def main():
    st.title("Movie Recommender System")
    query=st.text_input("Enter your query")

    diag= ''

    if st.button('Suggest Movies'):
        diag= recommendations(query, list_embedding,df)
        st.dataframe(diag)

   # st.success(diag)

if __name__=='__main__':
    main()    
