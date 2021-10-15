# CNN_LSTM_Text_Classification
# Medical Notes Classification

## Table of contents

1. [Introduction](#introduction)
	1.1. [Topic](#topic)
	1.2. [Dataset](#dataset)
	1.3. [Request](#request)
	1.4. [Technologies Used](#technologies-used)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model](#model)
	4.1. [EDA](#eda)
	4.2. [Solution](#solution)
	4.3. [Result](#result)
5. [Contact](#contact)

## Introduction

This is a final Project of ML course of VEF academy. 

### Topic 

Medical notes is the useful information source for patient data extraction. Notes classification is also an important task in Medical NLP domain. There are many techniques to solve this problem ranging from traditional method (Logistic Regression, SVM,...) to the state-of-the-art models (Transformer).

In this challenge, you need to classify the medical notes into five labels:

+ Surgery
+ Consult - History and Phy.
+ Cardiovascular / Pulmonary
+ Orthopedic
+ Others

### Dataset

Get data from this [Link](https://github.com/socd06/private_nlp/raw/master/data/mtsamples.csv). 

train_size = 0.6 ; test_size = 0.4; random_state = 0

### Request 

+ Code/result shoule be reproduceible
+ Metric to evaluate: f1_macro

### Technologies Used

+ [Pytorch](https://pytorch.org/)
+ [Numpy](https://numpy.org/)
+ [Scikit-learn](https://scikit-learn.org/stable/)
+ [Word2vec](https://pypi.org/project/gensim/)

## Installation

```bash
https://github.com/haiminh2001/CNN_LSTM_Text_Classification.git
```

## Usage

Execute project in 'Medical Notes Classification.ipynb'

## Model

### EDA 

#### 1. Labels Distribution

![image-20211015151003998](/home/kienanh/.config/Typora/typora-user-images/image-20211015151003998.png)

=> Unblanced

#### 2. Length of text

![image-20211015151416302](/home/kienanh/.config/Typora/typora-user-images/image-20211015151416302.png)

=> This data set has different lengths from each text. And it also has noise data - the data containing only 1 word

![image-20211015151813770](/home/kienanh/.config/Typora/typora-user-images/image-20211015151813770.png)

### Solution

#### Preprocessing & Encode text
1. Use library gensim.word2vec for encoding trainning data, each word is encoded to a vector (500,).

2. Split a text into segments, and we let it overlap itself.

   Example:

   It's a lovely day, let's go outside!

   \# Segment size = 2, segment overlapping = 1

   > [It's a, lovely day, let's go, outside]
   
   \# Segment size = 2, segment overlapping = 2
   
   > [It’s a, a lovely, lovely day, day let’s, let’s go, go outside]

3. Convert segments to tensor by tf-idf.

#### Architecture

![image-20211015155053184](/home/kienanh/.config/Typora/typora-user-images/image-20211015155053184.png)

We combine CNN and LSTM for this model:

+ Extract higher-level features (CNN)
+ Decrease dimensions of data (CNN)
+ Secure order of segments to feedforward (CNN + LSTM)

Loss function used: f1_macro score

### Result

![image-20211015155914399](/home/kienanh/.config/Typora/typora-user-images/image-20211015155914399.png)

Dealing with small, unbalanced and noisy data, we have reached 0.48 f1_macro score - a pretty good performance in comparision with 
0.38 of BERT model.

In other hand, we achieved 3 milestones of success:
+ reduce the influence of meaningless words: We found that in the data set, there are many words that have little meaning but appear many times. So we use 

  => We use tf-idf to solve this problem

+ balanced recall_score: When we started, we ran into unbalanced recall_score problem because of unblanced dataset.
    => We solved by adding weights to the loss function

+ f1_score increase stable: if we use the built-in loss functions of pytorch like categorical_crossentropy loss, our f1_score will increases unsteadily.
	=>  we made our own loss function: **loss  = 1 - f1_score** 

 <img src="/home/kienanh/.config/Typora/typora-user-images/image-20211015162217525.png" alt="drawing" style="height: 300px"/> <img src="/home/kienanh/.config/Typora/typora-user-images/image-20211015162139191.png" alt="drawing" style=" height: 300px"/> 




## Contact

Lê Trung Kiên         -    [kien.letrung610@gmail.com](mailto: kien.letrung610@gmail.com)
Nguyễn Hải Minh    -    [haiminhnguyen2001@gmail.com](mailto: haiminhnguyen2001@gmail.com)

Project Link: [https://github.com/haiminh2001/CNN_LSTM_Text_Classification](https://github.com/haiminh2001/CNN_LSTM_Text_Classification)



























