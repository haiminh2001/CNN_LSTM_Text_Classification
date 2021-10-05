import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

data = ['I like dog', 'I love cat', 'I interested in cat']

cv = CountVectorizer()

# convert text data into term-frequency matrix
data = cv.fit_transform(data)

tfidf_transformer = TfidfTransformer(use_idf = True)

# convert term-frequency matrix into tf-idf
tfidf_matrix = tfidf_transformer.fit_transform(data)
idf = tfidf_transformer.idf_
# create dictionary to find a tfidf word each word
word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

for word, score in word2tfidf.items():
    print(word, score)
