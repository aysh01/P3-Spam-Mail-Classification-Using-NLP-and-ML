import streamlit as stream
import pickle as pkl
import os
os.system('pip install -r requirements.txt')
          
with open('spam_model.pkl', 'rb') as model:
    spam_model = pkl.load(model)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer:
    tfidf_vectorizer = pkl.load(vectorizer)

stream.title('Spam Detector..')

user = stream.text_area('Enter Your Text Here..')

if stream.button('Predict'):
    if user.strip() == "":
        stream.warning('Please Enter Valid Text Message..')
    else:
        user.strip()
        user.strip().lower()
        user.encode('utf-8').decode('utf-8')

        input_vect = tfidf_vectorizer.transform([user])

        pred = spam_model.predict(input_vect)

        if pred[0] == 'spam':
            stream.error('Spam Message..')
            print('spam')
        else:
            stream.success('Not a Spam Message..')
            print('ham')
