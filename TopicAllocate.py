import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]',' ',text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()
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
    word2vec = Word2Vec(texts, min_count = 1, window =  window_size, size= self.vector_size)

    # create dictionar
    self.dictionary = list(word2vec.wv.vocab)
    self.w2v = word2vec.wv
    
  def doc2vec (self, text_data, window_size = 4, vector_size = 200, segment_size = 10, fit = False):
    self.vector_size = vector_size
    self.segment_size = segment_size
    #lemmertize texts 
    texts = lemmertize(text_data)
    text_matrix_size = np.amax(np.array([len(x.split()) for x in texts])) / self.segment_size
    if text_matrix_size == int(text_matrix_size):
      text_matrix_size = int(text_matrix_size)
    else:
      text_matrix_size = int(text_matrix_size) + 1
    #encode vocabulary to vectors
    if fit:
      self.cbow_fit(texts, window_size)
    cv = CountVectorizer()

    #calculate idf for each word
    data = cv.fit_transform(texts)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    word2idf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

    #transform texts into matrixs
    # ts2vec = np.zeros((len(texts), text_matrix_size, self.vector_size))
    ts2vec = []
    for textidx, text in enumerate(texts):
      words = text.split()
      #calculate tf for each text
      vectorizer = TfidfVectorizer()
      vector = vectorizer.fit_transform([text])
      text_dic = vectorizer.get_feature_names()
      tf = vector.todense().tolist()[0]
      word2tf = dict(zip(text_dic,tf))
      n = len(words)
      segments = [words[i : min(i+self.segment_size, n)] for i in range(0, n, self.segment_size)]
      
      text2vec = np.zeros((text_matrix_size, self.vector_size))  
      for segidx, seg in enumerate(segments):
        seg2vec = np.zeros((1, self.vector_size))
        known_size = segment_size
        for wordidx, word in enumerate(seg):
          tf_idf = 0
          try:
            tf_idf = word2tf[word] # * word2idf[word]
          except:
            known_size -= 1
            continue
          try:
            seg2vec += self.w2v[word] * tf_idf
          except KeyError:
            known_size -=1
            continue
          if known_size != 0:
            text2vec[segidx] = seg2vec  / known_size
          else:
            text2vec[segidx] = seg2vec = np.zeros((1, self.vector_size))
      # ts2vec[textidx] = text2vec
      ts2vec.append(text2vec)
    return ts2vec
