# Stock Predictions using neural networks
## Ignacio Fernández Adrados
##### UOC (iadrados@uoc.edu)
---


This notebook creates multiple RNN and LSTM neural network models for predicting the movements of the stock exchange shares and evaluates the acuracy of the predictions for determining the recomendation of buying or selling shares depending on multiple parameters used for training the different models. Apart from the price of the shares used for the training, it analyzes the improvement of the predictions including sentiment analisys based on finantial news.

The sets of tests used for the training is built with the following parameters [and examples]: 

+ intervals         ['1min', '10min', '60min', '6h', '12h', '1D']
+ algorithms        ['RNN', 'LSTM']
+ steps             [2, 5, 10, 20]
+ units             [50, 100., 150]
+ batchs            [16,64,128]
+ profits           [0.1,1,10,100,1000]
+ epochs            [5, 10, 20]
+ sentiment         [True, False]
+ useFinBERT        [True, False]
+ relevance         0.0-1.0
+ useIndicators     [True, False]

It includes 2 different sentiment analysis approaches:

1. The first one under the parameter **sentiment** is based in the sentiment datasets provided by Alpha Vantage (https://www.alphavantage.co/) wich apart from the header and the subject of the news, offers the calculated sentiment and the relevance of the calculation for every stock.

2. The test performed under the parameter **useFinBERT** is based on the pre-trained model found in https://huggingface.co/yiyanghkust/2finbert-tone created by {yang2020finbert,
    title={FinBERT: A Pretrained Language Model for Financial Communications},
    author={Yi Yang and Mark Christopher Siy UY and Allen Huang},
    year={2020},
    eprint={2006.08097},
    archivePrefix={arXiv},
    }
