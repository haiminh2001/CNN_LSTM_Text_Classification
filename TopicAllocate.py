import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]',' ',text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    translator = str.maketrans('', '', '_%')
    text = text.translate(translator)
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

  def cbow_fit (self, text_data, window_size = 4):
    texts = text_data
    #split into words
    texts = [text.split() for text in texts]

    #embeddind words
    word2vec = Word2Vec(texts, min_count = 1, window =  window_size, vector_size= self.vector_size)

    # create dictionar
    self.dictionary = sorted(list(word2vec.wv.key_to_index))
    self.w2v = word2vec.wv

  def cbow_w2v(self, word):
    try:
      return self.w2v[word]
    except:
      return np.zeros(self.vector_size)  

  def doc2vec (self, text_data, window_size = 4, vector_size = 200, segment_size = 10, fit = False):
      
    self.segment_size = segment_size
    #lemmertize texts 
    texts = lemmertize(text_data)
    
    #lean vocabulary
    if fit:
      self.vector_size = vector_size
      self.cbow_fit(texts, window_size)

    #calculate tf_idf
    vectorizer = TfidfVectorizer(token_pattern= r'([a-zA-Z0-9µl½¼ménièreºfü]{1,})')
    vector = vectorizer.fit_transform(texts)
    
    #transform texts into matrixs
    # ts2vec = np.zeros((len(texts), text_matrix_size, self.vector_size))
    doc2vec = []
    for idx,text in enumerate(texts):
      #get vocab in text and sort alphabetically
      words = sorted(list(set(text.split())))
      words = np.array(words, dtype = type('a'))
      
      #cbow_matrix with each row correspond to each word in cbow vector form
      cbow_matrix = np.array([self.cbow_w2v(word) for word in words])

      #calculate tf_idf 
      text_vector = np.array(vectorizer.transform([text]).todense().tolist()[0])
     
      
      #remove zero entries
      text_vector  = text_vector[text_vector != 0]
  
      #combine tf_idf with cbow by multiply each cbow vector by its tf_idf
      
      cbow_tfidf_matrix = np.diag(text_vector) @ cbow_matrix
      #remove zero rows
      cbow_tfidf_matrix = cbow_tfidf_matrix[np.any(cbow_tfidf_matrix, axis = 1)]
  
      #compress words into segments 
      n = cbow_tfidf_matrix.shape[0]
      
      if n == 0:
        t2v = np.zeros((1,self.vector_size))
      elif n <= segment_size:
        t2v = np.mean(cbow_tfidf_matrix[ : n], axis = 0).reshape(1, self.vector_size)
      else:
        end = n - segment_size
        t2v = np.vstack([np.mean(cbow_tfidf_matrix[i : i + segment_size], axis = 0) for i in range(0, end, segment_size)])
      
        #adjust rows remaining at the end of the matrix 
        if (n % segment_size) != 0:
          t2v = np.vstack((t2v, np.mean(cbow_tfidf_matrix[n - (n % segment_size) : n], axis = 0)))
      doc2vec.append(t2v)
     
    return doc2vec
