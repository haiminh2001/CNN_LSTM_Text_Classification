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

![image-20211015151003998](https://lh3.googleusercontent.com/fife/AAWUweX7dP7XbZpJfCk4oOYCJ4MzS-OU-fVR4H4jxEP6DX6LweOvC1e5Fm4rDYXJ7o3tCnNTLjhKel98ZGebse5JYE5qYZk4e5sIC7MFphLjEsOVtVxPcYePF6LQkqqPyTDX3ts49KAAWtJC6e5J6uMEqddUjYXqnAZtAJ8GfuREelKK1wIvn29bLcYVJDokrs6AvI9GhWjyyOIAo3uXXrwslxx4xKMhwr8SvOSmEKeZg1ZJT23S_UFhsZ4pM8bEqnO9sQTwBA12Vg_3Dc2e5rdbpwonZCgRjuH_awChWGR7LmAtU4hFBVjS3Km5oLIlZ8_AkYy0QOmjY2urF4yzPArWnLyr3UuTUp1ZmjUEUR8is75dmu2fHEglJ5rAFBTz3zKktqjOQYbQ8ot3zA6UD7q2L_KTuoN2mZWcSHRw-5X-vtnyXf1g807bPfni0lVlqKQjZvtCJqJDWbeUvr8p51RAbURmTqNflIPJQFp4pAtwBWWDLbmpxDN0m2fTAZ7yW6sgsJKXhSQW3bnJBvOzapwkB0OnIAhNWI64QE9RRKMar9OnE6jRVF9_jxGsd3sJ2COnj8zK8Zz5RAH9178GDxHaLd-mLGgEAvfChboITuCG-NePxaQvypVgvN8UHrP07d74l3JIfiYhNq48rvdbBYOGR40QWtBLC1zoWisbWXJyzjjIcm00GY3RHPuWFirChSi_Uc8i9M6m9QpM79v02jOUgNbl1VBbtV4L6ow=w1858-h948-ft)

=> Unblanced

#### 2. Length of text

![image-20211015151416302](https://lh3.googleusercontent.com/fife/AAWUweXCrj2OSRvR_P-yxSBANblMTI3El55FaEACifFtlBQI6Ywt69BEQqhTJQwFoXtr8TVvkJ_nMvISHnt9XerstCSP_dl04wDby80ItB-2Vu0i-wjYdLUzQpotRWjU_Pt051hT1W-Iw3xJZXjh6XIZJcKbqe9fAkbpwTBfZQMPP60f4FnSH2FcgvJ3yupaiu-qxzAP-1nc3mkvg0I0UBUm4eMHQOpaofNnlNsCTpQZbnX6uCgw8W61Js1Z7e2gifwEI1TuQsHv-6PK8l3g3h8UdvW98tNjKDGuBOX0-ksEgFKBJS6i6a6KYsom-fN-Mb0BXJVB5itbrMsgpj8-idAZkyLW7LTwZ0QxI_KBGRTwjFbehzXDAVL-IYyWh_PxcYH4GJpG1AwaEwbVY27-CZHm0KEcFzFw5yg4vJIhPA2Sg2JaX52n_cmmiLXsc-Cib2zPIyA-YGGYV0SBoyqx5CiChWN_etT16uiLq556DZk0Y2SEpbOew4iXlQKb9_MDBrBvqfMf9mRXxlqhyCugDAD4mwr0D1Rus6m90mP7OJD12wFdwC6vghRFyvGYC5ZEnCIPUO-uiIPjMIQTV9Ves8y1i36oE2FVMWN_KcM_V8lB5EHYg_o8xN0unWE12KG6sp3aTfyHYZDqvsbrzMyLiGyCJQE-jhOnIfYOFxHRIX_jICu-C3MZoibYb4Xxjot6iUuGx9MKa2x0IlFOV2nf6FwMoTaEzFUIozyT_c8=w1858-h391-ft)

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

![image-20211015155053184](https://lh3.googleusercontent.com/fife/AAWUweVYuTAADL-XQo4TLM_MiB5e9xJpQaI2-VnpF-t092i8foDUWF6OfwHx-j4SVkjwgp9fJH9sbi7TlAsxTmji7-InarRHTzUf9jQoekTh_vTbjf1DaXBQWsc5K5mLSQ4asSkqRH96o_5HphdeOW3Kl4RD-KVc-dvKv6fqq4_OyGsKTEdAWHeHysQqUpTS1AHOC3pxab_VAziUIV_h5uLFJh-hSaEdcQ-4enScAbZCfiH_tA45FDQkev6aPSvZP3xWuy41hk9HTyJoUDgVthUW20-EdGVk4Q6tcguY87hmTrxmAqijcr4oaOe88pa1S6GKozq-_fa5Ivc47WgnS2nzviiUoFBxXgd32LMKV0-wrRt_mDY_17Mu6EOPZ-_h6jZq7mgNVNLhNYT0UaLpYX95RDkJtovBS6wukGMDcxrCTV5gKKPofiF2PYR8c6HYkr2YsO_kZ2ZooOKKcLKXR8RVtZTpVuI6lFvQloGgqwzHYh-4I2hYkIHwcY-DdVeQHJJRw7E8-1RNlWWarjIxemLP7lu5PWCx3tlihfJehgMEJ9xiBRoiLWSze4r67OFXmTXFtp-xquS5UfGVftix2V3a2Y-SljoblJ327zRvh8WODr3Mfe1xoDGufycElzuVfct8P7M5v9ySxljSw8u1flNVAvCNbVHEwn9i1NYdOFG55lExq0SqK72bp7p4Chdaj0SERblGW1p3LOZ0JSUs-EzdUvdmsHrqGyGoeac=w1858-h391-ft)

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



 <img src="https://lh3.googleusercontent.com/fife/AAWUweW5JD0w9zGzLeYpQU48FD-Yy13yRLAA-wWqXHUF9k1iRYmm9P8zG26AIjaZl8GAgFB1qM1GKJMb9Y6wVvyocXp8gh8sMDHBSoVPj2dyPlca6S6AfYwN41q9G4q8oqEZlYHhISXqasfEiUArPxOiidsmhUV3QmFunsJ6okgWWTy3TQz57pxOsB6s4EFW_a806x8ro_OOcmQqz9yvdgykxa6XT9My_AISQM1Pf0tUQ0a0OuH3ZvR3nPtyrKSSJI6HzEQeA2spJK4spW942EdiFvgYcSB1qn7GSgQBhM5cAyANQdyuS0v6B5n9G69_qFy3qzaunS-qMCotOhRBOR1BPXOAQGmekau54osP1dR_OPQHIuVYJJss01kk3o6C_jFA-tM0llDFsVGlGdpjjkIb4YUgN8V0HLJx-qbfM9V7HMmnXcKiyhp4l6Zvt3ZjG14cNmp9R2LH9-S6DkXkNcfjIfE0-2ztvF-Jc0YuDyI9jc2yHuDqr7tp2MRSrmnVlckCWh_DvDBCfgp2QjkXDeWR3VAwAPzEazO4d7uoq7VsFMuSQDyFYbcQ9tfHvvUov_-r1czYxmwWV_0R0V8FlKYGg-ClhtU1pGlUROAt5HoTH-JiXBaB2XV0OLe0NQ2Pnb4XHOHIGyepxyBvQCrdC4QWhCj_nN7pWROmlphkuiiNzA9I227_e5WWCzwzav0bVHjQDzmM13xzUMAvBnvY2Bsw-Npz711wFfPHpuA=w1858-h391-ft" alt="drawing" style="height: 300px, width: 400px;"/> <img src="https://lh3.googleusercontent.com/fife/AAWUweWnp0BRjqV8VTNNgRzDUcc6U6r2ADBxtfGPytQ80JRY74XDywFui2KyCCjlQ6wReKg4Jw0ziHsow95dLrosf-EbKTD9gBEpULdMeYbcgn-Q4BcUsrOh4OGlUuOI7lo-enBTp26vAwgNu_ErBDrnbv7QRwtvhq7M3jRw4i-fC0e043IRj1eYuUdKrjYD3UkyBZrBeSzsOMvSkK1kJKu5QtFYE8O-1hGoM3Fadu3C8B-gzVyBisV_b48Ekft8SxdkCkxaVyIrEB7XGysGGmvdi_Hul8bzkL_BW2KvvwEmiEu5Q_IxGW4AJjGEolG0FHs0xMesEQT5YY57pt60ReIbE--V7eWJXtyRCP8jRs_HMi2_OxIK6aBOaU74eYt29p34ZTlTA5kVh2_8Elz-PtESBxcSghy5HMy9nw0BS8PpS19hR-2Wgfdva4vdnBwQAUmKLCcz6T5o4IGhyGlcXFzHxAzkY2HXRs0x-s1P4bjv2T0fcoGhopx-jFtgjdU5l3jBhIhLzwCNTjJdQmTR0lPhFERVya6g9F-A8shpJRfNYE2fmgRB8ftC78AwnOL0xiRf7YJzhzN_ZHIMeC8i1GpbMEcgf0FtgMSU44NhskV7z3WR33f9BXVQcxZ9m3piz7irwtdTM31AW5FhBBXeqaLleWPZlvoWVg4mPpcT1FteJx_PPQpovoWg3zFpU8iS84CR8wnH9d3rAa4w5RwrOlXx3f3CDTCQUYdrlN4=w1858-h391-ft" alt="drawing" style=" height: 300px, width: 400px;"/> 




## Contact

Lê Trung Kiên         -    [kien.letrung610@gmail.com](mailto:kien.letrung610@gmail.com)

Nguyễn Hải Minh    -    [haiminhnguyen2001@gmail.com](mailto:haiminhnguyen2001@gmail.com)



Project Link: [https://github.com/haiminh2001/CNN_LSTM_Text_Classification](https://github.com/haiminh2001/CNN_LSTM_Text_Classification)



























