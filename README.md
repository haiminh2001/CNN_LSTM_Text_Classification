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

### Libraries Used

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

![image-20211015151003998](https://lh3.googleusercontent.com/fife/AAWUweXGjtDhzOoqh6hArWt1awmPmV10LoJuurbHu9cks7oUkLjUzh8x1y1K6iPkUJS1QOX2J_4Ztq7QGZy0lkg8LMuAzP-rvFFhyHuOUrPpQxkE5ch-KlBkIVJ8SHCjSNqjXZRKXi5HR8KgB33TU2vEv8seBGNmNJlI9eAy2qm6_4BAoFrZuVhvnmQ57BTUIWncr_U_DKY2_Q0pi__4mlz9nUUqpAX1mZ32hb7ARA9BjMX_q14ljpDcDzbKV8gHpuLJ55N3o48iLtqH8M5rLWnxi7cJKXpX12PF07C_3Ql3_1fyWdyeNHiWh1v-Nncy3ySTCmeFjJceDvnCa-emO_KZwAq7Uc1gGJROTzYtxxd1ee74a1qfJoO1-DViKLHc6U282qpnRTIGTwwqlOH4EOL1sgMAIVjtL1bQY9NrZ8Ek9T940gJjwW7fevlt-ZUxESEczBzA77Ag9f5xvhU_ZuR774j-2i722x-2Jg4i_4pm-_8pCict4MzXn31CNWizFGB-7-0Beo_7ryARWesvYytb0YgZN2hM8qwD7DmV_bmQC6aXs3p2gJ0zhW9d4tObRAf5NR_KdzIYJCj--emRWWPPTR3J8Mi9xfKrabWdMUB5xWdz9LF8HmLxYM-1acn0sVkX1-m970b42WvH2f0tWZPLeGm39yMwvrCUb-0NpWEAXNklwKl4KYjIofG9ohhvEv49n56i1wfF8pUsbmlHsa4itgQx3W1qSSInISY=w1858-h886-ft)

=> Unblanced

#### 2. Length of text

![image-20211015151416302](https://lh3.googleusercontent.com/fife/AAWUweUNAUQhPIs3ylm037b8DGWr1j0n-tvEg1FZg1kVc0eW-hwtlvO4cFZ6MeU03j11ahE6QhsrC4r0JE1O2qAKaWkgKps6KhHIE52AUsiIX1ouqBVMoslDVufuzsJwkm-Tbn-qzhNsPkqsLxlqTRkSEAYJgdIf_L8M6JiRJdW15sTIaetvKaFbNhl0E7YGMbS9WPDa0Fat4ziP24KrIT3B-4nymyKISSZkjhEdkRP0OHyAtO9ZDfXHIXnE1BSfD19qHmyrN-C2hXdz7leefERQI5pmqQ9uyG-SKXy6YlWPiDCAK2ZLEqQEnOWGkN0Lv_8kdOaAXXG0T0Y__Rpo4HXvAKJWlFV3ThzLNget6a6CDQn3sMSSO5e08Q5kp7mGb30lFSt9eDIREWVXsh8Nr_AX9vBaFaJKcDNwDfhb_aaOC16YQugqxwsOa3iBk9P41eD-smAXuin56FJ4zeKJK8MdpmNdqrbd8B5cxeBJkzguM530_z3_DUC2uKiK8UoXL20eYJkIUtvb9VOjuW6XsJmZBFFgyroqftwyTtY4ru3xj9rM75cqbxgPA-MxGp4L3P-3cduSR60jORqLo4uT6B3yXvFTK2TJp6o2MQVF2RvJqFcoucD2bEN7KT72OaGGcebT527Q3AEJ4iDgnI9RGkYW6WvGb-OgA-G4OzO9w8CNyK_fxntHGnvtb3LYyRGu7vwgnRdS-_JeGC-dw-_mFAZb6coVSqEwBpZGP5Y=w928-h385-ft)

=> This data set has different lengths from each text. And it also has noise data - the data containing only 1 word

![image-20211015151813770](https://lh3.googleusercontent.com/fife/AAWUweVNmWSfq5IOUl5BC4UzYXaVAyiydthix9lfK8xsDn67mOBYKL4jtxBPMgTbmhjn_ZU3WWKQ6In4_Qoxq_-AhGr-L0uTJhCYQ-OYPiXHco2AeWTDEFnVDfMVcuJC3NAyuMMTnTB7eAWJHgQs8Dn9jFeMZDML-OMKYj0NjJBeH7RbD1dpwqjmP05Bl-JXmW58nUxjXalPfMfDpI0u9PeCyh50Fu5gKntrkFGFh6VfgOCYgNju9GKEI4aolGnAFVOb0Css7mdhfvodseRXCGydIyRl96HxhHmJRzceKQsUk7LB_BPeAe0t1h7V2JcrIj9fayw8vytqdZkBiq5AHiKgDUHU3yn8HfOePUJd3FMtnSPulGdS1bliQ2faBmpMl-xHi_SalwvWWGV8BgxTt_AwkOPDmC9L-FrXQU-ixEL_JR4xjLMfpD03KclXRxcZ-BwRzGi-1RLLyRK7CsJFzft8yQhBhw4ESJENR1VLNlbJNhX0e5QYCLQa4dBXDaHCBFGE6RRujbXDt8hZVRipiHlYo4vd0VbLkYjs0F0Lsn_Pbuv-vHov6YFXbIhdgUetczlbmgd2h-l6Szlgr5BTNuV2E3cw2AKftluu88S83bW4yK-DHl8OUm1xIab4w8_BIfL7m0JkJ69rgrQt3OwQW9Yx6msHpgWEh8pyLwEWna0yPLTXuc-beqDrdJAoSULRbqurAiBJ1Se7KeVqShniBAGcOHWW7Cv_wWtVkck=w928-h916-ft)

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

![image-20211015155053184](https://lh3.googleusercontent.com/fife/AAWUweUOfEUWbAFzgTBe63S5rbo2lFFVVqvKi8kd4BblCfFNjCRSdoc0ckobyOCcVuJVP5D9PEMyQkS6RIgyGywe2aBe82h20XfjeMdu22oMj-znKz8eBOBlbtaC9LPXX9_6Lr8T4lNeD4Bsfgd-KrJMe98yvmLAlWB8PjFOVTUZJCdlio4GONbJmIXS-Wo7hJ7M6O1OA4BDNMew0GLHDPWChOhwSkgCBd8Q_L2_z1XAA_N2KeEkqT38inAMDWv-MHBrwYmkoUrRbsbs4dGkTF7Jia5B6uDzU_ovur4VY1j5jDLE8uN_OpMecQ36A-KsJ-ZukZ9NT5-qFePt1ahYtEMAyznuoBsqXRgqVkDbNytRl9NuXyaRTfbd5OTkYp3iSA_6U-b80SJDnFIwbcySgjovzdNxEqAzT1eTj1dQQAP1NdbVZhwSCgnj9iqbDDMcdEAFbyRSCKpNjEzScOEiXDlES9PppdO12ns5g39sts0AlDTOz4fQP7F4mH_2jEpknPxymePE_XYUgtcvqR3OoI5gI7XCY3BGuJXv7FaWXMBMckXchZ6HbBW4EbxoYu22oGnmNBzsmJHQMkXeezqvFueIKmRpRtgGaHB9r06nbjb7y_xe2JeMIKQY1GJRL3h2ykXIkGYAl5Bsh86pFFvKOkHRLIRY3o7a2xazVRMx0-s1tL2lfFBDEazCCEwe3tKbtdhwbetgTMu79Zd-wJzB7n79h4CeHPfPGby9Mfc=w1858-h948-ft)

We combine CNN and LSTM for this model:

+ Extract higher-level features (CNN)
+ Decrease dimensions of data (CNN)
+ Secure order of segments to feedforward (CNN + LSTM)

Loss function used: f1_macro score

### Result

![image-20211015155914399](https://lh3.googleusercontent.com/fife/AAWUweVoWouTxkbYz71CQg1-mzevbn-JXfjKvzD5if_0zTSMZ5GZpBRn7bvaiuvjTYDWrEuA6D3IhDbH0xQZWHwUIMPePNas0S-aJNFLijBG_jQm89AHKNC4_k4hHt1JL0PtcV4N935yDrkFS2AyDwGuJCo0qsgfG_ECeyqq-gK9HcOSVzJbKQt7gIfHoq4uPO2__faj7IPk2AAS7cpgwZe5YutwU5VAEQCqENU-JxLkQIckOtnjG8WE5VS2ubvP-VH-03ytbARcUeOb33ARG8KWV8eZ8PwCWsLkgR3CoRV77_OB1Og5m9V9cBHfFX7kvP8PIsxyjGCCvnQ1LU1PDscn_celeBdlVbrNace90FkaSa9NiZ4TyCbpS1cTmvFbWh-F1J_TmWPsH0u_CKSnVYqzwijJlA_CIrRPVSBKFn7pevAOEG-oCaoM-s8WvWxAgMPG4vZKcq3MhZvGq9f-YCuL6eebvD4f-oLZWZX_FlBNC6pYS4YjrEIu0pe2zavEbtpvL4WlHJgTuFiklpwPbUcoZjB76JOiqNSGOsVlICuxaJVlY7w0ZO4-aAV4vW2uq423t4shg22ryEechGdpSZt07IDkLlYN57jszSjW355OvSmIj2SID3dOHmlcYMJh72vjMR5Yf8szACcGe5uu5oqzpHEnP7TWFN04iqlbfmvlsL0Jzr0p4yfaUXwf2DCNmxRKBcyMBprlfjZQRSRUxZEqrgEo-B9i7vf6w3I=w928-h385-ft)

Dealing with small, unbalanced and noisy data, we have reached 0.48 f1_macro score - a pretty good performance in comparision with 
0.38 of BERT model.

In other hand, we achieved 3 milestones of success:
+ reduce the influence of meaningless words: We found that in the data set, there are many words that have little meaning but appear many times.

  => We use tf-idf to solve this problem

+ balanced recall_score: When we started, we ran into unbalanced recall_score problem because of unblanced dataset.

  => We solved by adding weights to the loss function

+ f1_score increase stable: if we use the built-in loss functions of pytorch like categorical_crossentropy loss, our f1_score will increases unsteadily.

	=>  we made our own loss function: **loss  = 1 - f1_score** 



 <img src="https://lh3.googleusercontent.com/fife/AAWUweVkR_FLxKHag6pujz46EOylelZVhu4qUgcjNpLn8RQbMcbaeX0B1TpmUglrhPfvN9QAOi71XWZIrNC3oMwtrIpOB5H0O_ftvTAlgMsJfx_ErGrALxQ5n2jfPdPxg6AN_ywL-O7HwYuBGQ2oYzoiwiHhJZqyXneeaaGBS1_ia8K-_xTz4TJB2qrqQikBY0oZR45x8ZjqP8fovgNGEzYj7g2lQjdCGEAMV_KACBm9ASlqfSBdWDVCALZxb7WMNberiksQdgiO-FfI57vJV-eY3jZV11KeU8KpJiLSiAbTixhho0wNyoSIHL84lxvcyLLW9Tr5umiInplDoIrVkRgKmwiMxhVrHXOhgA3CJCw2A3g-Vzag8C4eQpxC3i0dfL5I8cxypNX_AJ_R_yrdEw-sEaVP3YR58-423ErrVrmeoulQloMeq6tGZF49fenNY-nqHwneG-y0zWLrFELlRN26zzAZoSHhBUSRiid5QnxfoscQv7_HcUfB0i1DSstLNHyB3n3DvtJbvBjkcOzEZEpTPOToAeNSEOw8QivNkT37fXUQZCAgxlkU-fkQcX2Bq-tJ-XTl3szIxMaW5B4TcIMdSmOxywwZL0jA6ExRV-oDEATUvuoyyAoL0yo5yjl3WG5nOaSFm8_DyT56RB26tk1sgIOR4uHB1BGVL_ivWxpp1q29y-UAE1-S0t_2mBfPngiUxurwW1dgZqE1RE7zta9JOzKf2wJwGMB6eEc=w928-h385-ft" alt="drawing" style="height: 300px, width: 400px;"/> <img src="https://lh3.googleusercontent.com/fife/AAWUweW0BOtAmgsB9hof4gdSrJsRa3wA98fH2w4WoBNdRn8WKXQtx3YTEV2roszH1fmfArgRydcdfo3R5Cu1ylTYxJcZ4rYkRRlZqLmB4O-lce0WWEtnfotVibhTgMaYqRBlKgFhx7MsefVZMg9kUMv5BjPNsaEMG77GnRxauk1wDTRW7CSrflUF7sX3AzeXBM6zCOygLKgUyYZ9AzWLT7OOcv86iLcbq1PMjXCyVISsFvbbiFGQFUkdxkF8mLy_K_zec1FUU2I8AGBmU_OOwqhUePJIp2lPx1wbyzF3JjU7ChF1vkoz1L-MZJrx_kXf4LlA1ppyGaKCtFAHstN3ilgf_TET_EvK42GssChJ6JTVU8pVl3gH2ejICwMWRYU8lguLHh2hAP_JGn_166kb14nmTRltVq0CqYLnM3sPpUqsUVxVKmALSW7XzaNLQAIlAwz3DhF0CWukUAj_AxsaszPUQuaGupy7O8BnJCteDZk3__1glWWK_r5cP7atC49MlOkfeYTqADX3eEE4NbX84rd0LAFgpkCYrpcpzv_1F4NRm0x7FYBYapVkkVKMrNYD5CSUR2w_EDaywpNT6ByQf46kgVg6gWwOCZx9gTS_ZXI8QqENUOrJ8S5P9-U_5QV7UnT6aaXq09W6jRBuKSwehpvUtPH3AkYUI1iHCN5JYyDqCtOmYnZXBu7IBYfwzMNQ9Oa_1fnSIPvLmtCWaF5sKncpCbQUpIYqTY--Q2I=w1858-h391-ft" alt="drawing" style=" height: 300px, width: 400px;"/> 




## Contact

Lê Trung Kiên         -    [kien.letrung610@gmail.com](mailto:kien.letrung610@gmail.com)

Nguyễn Hải Minh    -    [haiminhnguyen2001@gmail.com](mailto:haiminhnguyen2001@gmail.com)



Project Link: [https://github.com/haiminh2001/CNN_LSTM_Text_Classification](https://github.com/haiminh2001/CNN_LSTM_Text_Classification)



























