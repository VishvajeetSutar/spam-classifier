import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()



def transform_text(text):
    # 1.lower case
    text=text.lower()
    text=nltk.word_tokenize(text)
    
   #tokenization 
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    #stopords and punctuations
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
     
    #steamming
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return ' '.join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/Spam Classifier")


input_sms=st.text_input('Enter the Message')

if st.button('Predict'):
    #1.preprocessing:

    transformed_sms= transform_text(input_sms)



    #2.vectorize:
    vector_input= tfidf.transform([transformed_sms])
    


    #3.predict:
    result= model.predict(vector_input)

    #4.display:
    if result == 1:
        st.header("Spam")
        
    else:
        st.header("Not a Spam")
        








