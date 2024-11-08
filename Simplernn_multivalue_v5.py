import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, SimpleRNN, Input, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

FILE='AMAZON.csv'
#FILE='/content/drive/MyDrive/Colab Notebooks/AMAZON.csv'

file = pd.read_csv(FILE, sep=';')
file=file.loc[::-1].reset_index(drop=True)

N=60     #Minutos agrupados
DATEFILTER='' #Filtro de fecha


df = file[file['Date'].str.startswith(DATEFILTER)]
df=pd.DataFrame(df)
df=df.drop(columns=["Date"])
dfgroup=pd.DataFrame(df['open'].groupby(df.index//N).first())
dfgroup=dfgroup.join(pd.DataFrame(df['close'].groupby(df.index//N).last()))
dfgroup=dfgroup.join(pd.DataFrame(df['low'].groupby(df.index//N).min()))
dfgroup=dfgroup.join(pd.DataFrame(df['high'].groupby(df.index//N).max()))
dfgroup=dfgroup.join(pd.DataFrame(df['volume'].groupby(df.index//N).sum()))
dfgroup = dfgroup.reset_index(drop=True)
dfgroup.info()

def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]

dfgroup['EMA7']= dfgroup['close'].ewm(span=7, adjust=False).mean()
dfgroup['MACD']= dfgroup['close'].ewm(span=12, adjust=False).mean()- dfgroup['close'].ewm(span=26, adjust=False).mean()
dfgroup['SignalMACD'] = dfgroup['MACD'].ewm(span=9, adjust=False).mean()
dfgroup['RSI'] = get_rsi(dfgroup['close'], 14)
dfgroup = dfgroup.dropna()
dfgroup = dfgroup.reset_index(drop=True)
dfgroup.head()

#Normalise data into (0,1) range
normData=dfgroup
scaler = MinMaxScaler(feature_range=(0, 1))
#normData[['open']] = scaler.fit_transform(normData[['open']])
normData[['open','close','low','high','volume','EMA7','MACD','SignalMACD','RSI']] = scaler.fit_transform(normData[['open','close','low','high','volume','EMA7','MACD','SignalMACD','RSI']])
normData.head()

plt.figure(figsize=(14,4))
plt.title("Dataset 'Close' data ")
plt.plot(normData['close'],color='black')
#plt.scatter(normData.index, np.where(normData['buysell'] == -1,normData['close'], None), color="red", marker="v")
#plt.scatter(normData.index, np.where(normData['buysell'] ==  1,normData['close'], None), color="blue", marker="^")
plt.show()

"""### Split the values in train and test

So, we took only 25% of the data as training samples and set aside the rest of the data for testing.

Looking at the time-series plot, we think **it is not easy for a standard model to come up with correct trend predictions.**
"""

S=0.7
step = 5

split = int(len(normData) * S)
#values = normData.values
#print(values)
train = normData[:split]#.drop(['buysell'],axis=1)
test = pd.concat([train.tail(step),normData[split:]]).reset_index(drop=True)

print("Train data length:", train.shape)
print("Test data length:", test.shape)

plt.figure(figsize=(14,4))
plt.title("Dataset 'Close' data split")
plt.plot(normData.index.values,normData['close'],c='black')
plt.axvline(normData.index[split], c="r")
#plt.scatter(normData.index, np.where(normData['buysell'] == -1,normData['close'], None), color="red",  marker="v")
#plt.scatter(normData.index, np.where(normData['buysell'] ==  1,normData['close'], None), color="blue", marker="^")
plt.show()

"""### Converting to a multi-dimensional array
Next, we'll convert test and train data into the matrix with step value as it has shown above example.
"""

def convertToMatrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step
        #print(i, d, data[i:d)
        X.append(data[i:d])
        Y.append(data[d,1])
    return np.array(X), np.array(Y)

trainX,trainY =convertToMatrix(train.to_numpy(),step)
testX,testY =convertToMatrix(test.to_numpy(),step)
#print(trainY)
print("Training data shape:", trainX.shape,', ',trainY.shape)
print("Test data shape:", testX.shape,', ',testY.shape)

"""### Keras model with `SimpleRNN` layer

- 256 neurons in the RNN layer
- 32 denurons in the densely connected layer
- a single neuron for the output layer
- ReLu activation
- learning rate: 0.001
"""

UNITS = 500 #num_units: Number of units of a the simple RNN layer
DENSEUNITS = 32 #Number of neurons in the dense layer followed by the RNN layer
LR = 0.001 #Learning rate (uses RMSprop optimizer)

model = Sequential()
model.add(Input((step, trainX.shape[2])))
model.add(LSTM(units=UNITS, activation="tanh",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=UNITS//2, activation="tanh"))
model.add(Dropout(0.2))
#model.add(SimpleRNN(units=UNITS//3, activation="relu"))
model.add(Dense(DENSEUNITS, activation="tanh"))
model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer=RMSprop(LR),metrics=['mse'])
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
model.summary()

"""### Fit the model"""

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        #if (epoch+1) % 10 == 0 and epoch>0:
            print("Epoch number {} done".format(epoch+1))

batch_size=64
num_epochs = 5

model.fit(trainX,trainY,
          epochs=num_epochs,
          batch_size=batch_size,
          callbacks=[MyCallback()],verbose=0)

"""### Plot loss"""

plt.figure(figsize=(8,3))
plt.title("RMSE loss over epochs",fontsize=16)
plt.plot(np.sqrt(model.history.history['loss']),c='k',lw=2)
plt.grid(True)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Root-mean-squared error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

"""### Predictions
Note that the model was fitted only with the `trainX` and `trainY` data.
"""

trainPredict = model.predict(trainX)
testPredict= model.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)
print(trainPredict.shape)
print(testPredict.shape)

plt.figure(figsize=(14,4))
plt.title("This is what the model predicted",fontsize=18)
plt.plot(testPredict,color='blue')
plt.show()

"""### Comparing it with the ground truth (test set)"""

OFFSET=0

index = normData.index.values
plt.figure(figsize=(14,4))
plt.title("Ground truth and prediction together",fontsize=18)
plt.plot(normData['close'].iloc[split+OFFSET:].reset_index(drop=True),color='black', label='Ground Truth')
plt.plot(testPredict[OFFSET:],color='blue', label='Prediction')
plt.legend()
plt.show()

PROFIT=0.01
OFFSET=0

decision=test['close'].iloc[step:]
decision = decision.diff()
decision = decision.dropna()
decision = np.where(abs(decision)<PROFIT,0,np.sign(decision).astype('int'))
decision = pd.DataFrame(data={'buysell':decision})#.drop(0).reset_index(drop=True)

predictedDecision = pd.DataFrame(data={'buysellPredicted':testPredict[:,0]})
predictedDecision = predictedDecision.diff()
predictedDecision = predictedDecision.dropna()#.reset_index(drop=True)
predictedDecision = np.where(abs(predictedDecision)<PROFIT,0,np.sign(predictedDecision).astype('int'))
predictedDecision = pd.DataFrame(data={'buysellPredicted':predictedDecision[:,0]})

index = normData.index.values
plt.figure(figsize=(14,4))
plt.title("Ground truth and prediction together",fontsize=18)
plt.plot(normData['close'].iloc[split+OFFSET:].reset_index(drop=True),color='black')
plt.plot(testPredict[OFFSET:],color='blue')

x = decision.iloc[OFFSET:].reset_index(drop=True).index
y = normData['close'].iloc[len(normData)-len(predictedDecision.index)-1+OFFSET:-1].reset_index(drop=True)

plt.scatter(x, np.where(decision['buysell'].iloc[OFFSET:] == -1,y, None), color="red",  marker="v", label='Ground Truth Sell')
plt.scatter(x, np.where(decision['buysell'].iloc[OFFSET:] ==  1,y, None), color="blue", marker="^", label='Ground Truth Buy')

plt.scatter(x, np.where(predictedDecision['buysellPredicted'].iloc[OFFSET:] == -1,y-0.015, None), color="orange",  marker="v", label='Prediction Sell')
plt.scatter(x, np.where(predictedDecision['buysellPredicted'].iloc[OFFSET:] ==  1,y+0.015, None), color="green",  marker="^", label='Prediction Buy')
plt.legend()
plt.show()

OFFSET=0

index = normData.index.values
plt.figure(figsize=(14,4))
plt.title("Ground truth and prediction together",fontsize=18)
plt.plot(normData['close'].iloc[split+OFFSET:].reset_index(drop=True),color='black')
#plt.plot(testPredict[OFFSET:],color='blue')

x = decision.iloc[OFFSET:].reset_index(drop=True).index
y = normData['close'].iloc[len(normData)-len(predictedDecision.index)-1+OFFSET:-1].reset_index(drop=True)

plt.scatter(x, np.where((decision['buysell'].iloc[OFFSET:] !=  predictedDecision['buysellPredicted'].iloc[OFFSET:]) & (predictedDecision['buysellPredicted'].iloc[OFFSET:] != 0),y, None), color="red", label='Ground Truth Buy')
plt.scatter(x, np.where((decision['buysell'].iloc[OFFSET:] == predictedDecision['buysellPredicted'].iloc[OFFSET:]) & (decision['buysell'].iloc[OFFSET:] != 0),y, None), color="green",  label='Ground Truth Sell')

plt.legend()
plt.show()

print("Predicciones no coincidentes con el conjunto de pruebas: ", np.count_nonzero(np.where((decision['buysell'].iloc[OFFSET:] !=  predictedDecision['buysellPredicted'].iloc[OFFSET:]) & (predictedDecision['buysellPredicted'].iloc[OFFSET:] != 0),y, None)))
print("Predicciones coincidentes con el conjunto de pruebas: ", np.count_nonzero(np.where((decision['buysell'].iloc[OFFSET:] == predictedDecision['buysellPredicted'].iloc[OFFSET:]) & (decision['buysell'].iloc[OFFSET:] != 0),y, None)))