import nltk
from gensim.models import Word2Vec
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from CNNLSTM import Train, Predict, CNNLSTM, f1_loss, TrainDataset

nltk.download('wordnet')
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

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
def fill_zeros(x, Vector_size, Max_size):
    try:
        missing = Max_size - x.shape[0]
        fill_in = np.zeros((missing, Vector_size))
        return np.vstack((fill_in, x))
    except:
        return np.zeros((Max_size, Vector_size))  
class Topic_Allocate():
  def __init__(self, vector_size = 500, segment_size = 20, segment_overlapping = 3):
    self.vector_size = vector_size
    self.segment_size = segment_size
    self.segment_overlapping = segment_overlapping

  def cbow_fit (self, text_data, window_size = 4):
    texts = text_data
    #split into words
    texts = [text.split() for text in texts]

    #embeddind words
    word2vec = Word2Vec(texts, min_count = 1, window =  window_size, vector_size = self.vector_size)

    # create dictionar
    self.w2v = word2vec.wv

  def cbow_w2v(self, word):
    try:
      return self.w2v[word]
    except:
      return np.zeros(self.vector_size)  

  def doc2vec (self, text_data, window_size = 4, vector_size = 200, segment_size = 10, segment_overlapping = 1, fit = False):
      
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
      if segment_overlapping > segment_size:
        segment_overlapping = 1
        print('segment_overlapping cannot be greater than segment_size')
      
      if n == 0:
        t2v = np.zeros((1,self.vector_size))
      elif n <= segment_size:
        t2v = np.mean(cbow_tfidf_matrix[ : n], axis = 0).reshape(1, self.vector_size)
      else:
        step = int(segment_size / segment_overlapping)
        end = n - segment_size
        t2v = np.vstack([np.mean(cbow_tfidf_matrix[i : i + segment_size], axis = 0) for i in range(0, end, step)])
      
        #adjust rows remaining at the end of the matrix 
        if (n % segment_size) != 0:
          t2v = np.vstack((t2v, np.mean(cbow_tfidf_matrix[n - (n % segment_size) : n], axis = 0)))
      doc2vec.append(t2v)
     
    return doc2vec

  def train(self, data, labels, batch_size = 16, epochs = 30):
    #vectorize data
    X_train = np.asarray(self.doc2vec(data, vector_size = self.vector_size, segment_size = self.segment_size, segment_overlapping = self.segment_overlapping, fit = True))

    #one hot encode
    self.onehot_encoder = OneHotEncoder(sparse=False)
    Y_train = self.onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))
    
    #transform X into sequence form
    self.max_size = np.amax(np.array([x.shape[0] for x in X_train])) 
    func = lambda x: fill_zeros(x, self.vector_size, self.max_size)
    X_train = np.array([func(x) for x in X_train])

    #create classifier
    torch.manual_seed(42)
    self.classifier = CNNLSTM(len(self.onehot_encoder.categories_[0]), self.vector_size, 2).cuda()
    #calculate amount ratio of each labels for weighting
    ytrain = np.argmax(Y_train, axis = 1)
    ratio = [(ytrain.shape[0] / np.sum(np.where(ytrain == x,1,0)) ) for x in range(5)]
    ratio = ratio / np.sum(ratio)
    loss_fn = f1_loss(weight = ratio)
    loss_fn = f1_loss(ratio)

    #create data loader
    train_set = TrainDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_set, batch_size= batch_size, shuffle = True)
    #train
    Train(epochs, model= self.classifier,loaders= train_dataloader ,loss_func=  loss_fn,lr= 0.001,wd= 1e-4,X_train_sequence= X_train,Y_train= Y_train)

  def predict(self, test):
    #vectorize test
    X_test = np.asarray(self.doc2vec(test, vector_size = self.vector_size, segment_size = self.segment_size, data_enrichment = self.data_enrichment))

    #transform into sequence
    func = lambda x: fill_zeros(x, self.vector_size, self.max_size)
    X_test = np.array([func(x) for x in X_test])

    return Predict(self.classifier, X_test)

