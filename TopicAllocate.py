import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk import tokenize

def preprocess(text):
    text = text.lower() # Lowercase
    text = text.replace('.','O')
    text = re.sub(r'[^\w\s]',' ',text) # Remove punctuation
    text = text.replace('O','.')
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

nltk.download('wordnet')
nltk.download('punkt')
def lemmertize(texts):
   #texts input type: list of string
   wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
   lemmertize_texts = []
   
   for text in texts:
    words = preprocess(text).split(' ')
    lemmertize_texts.append(' '.join([wordnet_lemmatizer.lemmatize(word,pos ='v') for word in words]))

   #return lemmertized texts
   return lemmertize_texts

class Topic_Allocate():
  def __init__(self):
    self = self

  def cbow_fit (self, text_data, window_size):
    texts = text_data
    #split into words
    texts = [text.split() for text in texts]

    #embeddind words
    word2vec = Word2Vec(texts, min_count = 1, window =  window_size, vector_size= self.vector_size)

    # create dictionar
    self.dictionary = list(word2vec.wv.key_to_index)
    self.w2v = word2vec.wv
    
  def doc2vec (self, text_data, window_size = 4, vector_size = 200):
    self.vector_size = vector_size

    #lemmertize texts 
    texts = lemmertize(text_data)
    
    #encode vocabulary to vectors
    self.cbow_fit(texts, window_size)
    cv = CountVectorizer()

    #calculate idf for each word
    data = cv.fit_transform(texts)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    word2idf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

    #transform texts into matrixs
    ts2vec = []
    for text in texts:
      sentences = tokenize.sent_tokenize(text)
      sentences = list(filter(None, sentences)) #remove blank strings
      text2vec = np.empty((len(sentences),self.vector_size))  
      for idx, sent in enumerate(sentences):
        sen2vec = np.zeros((1, self.vector_size))
        
        #calculate tf for each sentence
        vectorizer = TfidfVectorizer()
        try:
          vector = vectorizer.fit_transform([sent])
        except:
          text2vec[idx] = np.zeros((1,self.vector_size))
          # print('Blank sent ', sent)
          continue
        sent_dic = vectorizer.get_feature_names()
        tf = vector.todense().tolist()[0]
        known_size = len(sent_dic) #known_size is the numbers of known vocabulary words
        for wordidx, word in enumerate(sent_dic):
          tf_idf = 0
          try:
            tf_idf = tf[wordidx] / word2idf[word]
          except:
            known_size -= 1
            continue
          try:
            sen2vec += self.w2v[word] * tf_idf
          except KeyError:
            known_size -= 1
            continue
        if known_size == 0:
          text2vec[idx] = np.zeros((1,self.vector_size))
        else:
          text2vec[idx] = sen2vec / known_size
      ts2vec.append(text2vec)
    return ts2vec
