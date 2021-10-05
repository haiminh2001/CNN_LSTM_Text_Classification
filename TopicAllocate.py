import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
def preprocess(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\d\s]+', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

nltk.download('wordnet')
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

    # create dictionary
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
    for textid, text in enumerate(texts):
      sentenes = text.split('.')
      text2vec = np.empty((len(sentenes),self.vector_size))  
      for idx, sent in enumerate(sentenes):
        sen2vec = np.zeros((1, self.vector_size))
        
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform([sent])
        sent_dic = vectorizer.get_feature_names()
        tf = vector.todense().tolist()[0]
        known_size = len(sent_dic)
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
        text2vec[idx] = sen2vec
      ts2vec.append(text2vec / known_size)
    return ts2vec
#minh ngu ...