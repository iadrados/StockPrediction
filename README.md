# Stock Predictions using neural networks
## Ignacio Fernández Adrados
##### UOC (iadrados@uoc.edu)
---

This project demonstrates how the neural networks can help to determine a recommendation for selling or buying shares in the stock markets. It creates a set of tests for generating multiple RNN and LSTM neural network models with different parameters for predicting the movements of the shares in the stock markets. It evaluates the acuracy of the predictions. Additionally it provides two different aproaches to the analysis of the sentiment of the finantial news.


## Stock_Prediction_Models.ipynb
This is the main notebook for running all the tests and create he different neural networks models. The sets of tests used for the training of the models is built with the following parameters [and examples]: 

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

2. The test performed under the parameter **useFinBERT** is based on the pre-trained model published in https://huggingface.co/yiyanghkust/2finbert-tone created by Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." Contemporary Accounting Research (2022).


## Stock Exchange News Datasets
Downloads and merges the stock exchange news datasets from Alpha Vantage using the demo API Key. Replace the 'demo' key by a new key obtained in https://www.alphavantage.co/.
Specify the FILE_PATH, SYMBOL and YEAR for getting the required data

## Stock Exchange Datasets
Downloads and merges the stock exchange datasets from Alpha Vantage using the demo API Key. Replace the 'demo' key by a new key obtained in https://www.alphavantage.co/.
Specify the FILE_PATH, SYMBOL and YEAR for getting the required data
