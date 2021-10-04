import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re

def preprocess(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\d\s]+', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

nltk.download('wordnet')
class Topic_Allocate():
  def __init__(self):
    self = self
  def lemmertize(self,texts):
   #texts type: list of string
   wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
   lemmertize_texts = []
   
   for text in texts:
    words = preprocess(text).split(' ')
    lemmertize_texts.append(' '.join([wordnet_lemmatizer.lemmatize(word,pos ='v') for word in words]))
   return lemmertize_texts
   
  def bagOfWords (self, text_data):
    #convert to list of string
    texts = self.lemmertize(text_data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # create dictionary
    self.dictionary = list(tokenizer.word_index.keys())

    # create bag of words
    self.bow = tokenizer.texts_to_matrix(texts, mode = 'count')
  
  def cbow_fit (self, text_data, window_size, vector_size):
    self.vector_size = vector_size

    #convert to list of string
    texts = self.lemmertize(text_data)

    for i in range(len(texts)):
      texts[i] = texts[i].split()

    word2vec = Word2Vec(texts, min_count = 1, window =  window_size, vector_size= self.vector_size)

    # create dictionary
    self.dictionary = list(word2vec.wv.key_to_index)

    # self.w2v['minh'] = [..., ..., ...]
    self.w2v = word2vec.wv
    

  
  def doc2vec (self, text_data):
    #convert to list of string
    texts = self.lemmertize(text_data)

    ts2vec = []

    for text in texts:
      text2vec = [0] * self.vector_size
      for word in text:
        try:
          text2vec += self.w2v[word]
        except KeyError:
          continue
      ts2vec.append(text2vec)
      
    return ts2vec
