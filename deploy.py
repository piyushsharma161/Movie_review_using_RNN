#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Others
import re
import nltk
import string
import pickle

from nltk.corpus import stopwords


# In[2]:


def clean_text(text):
    
    ## Convert words to lower case and split them
    #text = text.lower().split()
    text = text.lower()
    ## Remove stop words
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 2]
    
    #text = " ".join(text)
    
    

    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Remove puncuation
    # pu = set(string.punctuation)
    # text = ''.join(ch for ch in text if ch not in pu)
    text = text.translate(str.maketrans(' ',' ',string.punctuation))

    ## Stemming
    text = text.split()
    text = [w for w in text if  len(w) >= 2]
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return [text]


# In[3]:


### load tokenizer
vocabulary_size = 5000
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))


# In[4]:


maxlen=500


# In[5]:


#And load it back, just to make sure it works:
model = load_model('../save_models/word_embedding_self2.h5')


# In[6]:


def predict():
    in_text = input('')
    if in_text in ['', ' ']:
        return 'Enter valid reveiw'
    text_processed = clean_text(in_text)
    sequence = tokenizer.texts_to_sequences(text_processed)
    pad_sequence = pad_sequences(sequence, maxlen=500)
    pred_probability = model.predict(x=pad_sequence)
    if pred_probability >= 0.5:
        return 'You liked the Movie'
    else:
        return "You didn't like the Movie"


# In[ ]:




