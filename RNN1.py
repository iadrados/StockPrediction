import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pandas import read_csv
import numpy as np
import math
from keras.models import Sequential # type: ignore
from keras.layers import Dense, SimpleRNN # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Read data from given url and extract the second column
def read_data(url):
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
#Normalise data into (0,1) range 
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    return data, n

def get_train_test(split_percent, data):
    n = len(data)
    split = int(n * split_percent)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data

#Reshape data into input-output pairs with specified time steps
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
#Prepare Training and testing data
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y

#Define the RNN model
def create_RNN(units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=input_shape, 
                        activation=activation[0], return_sequences=True))
    model.add(Dense(dense_units, activation=activation[1]))
#Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#Get error of predictions to evaluate it
def print_error(trainY, testY, train_predict, test_predict): 
    train_predict = train_predict.reshape(-1)
    test_predict = test_predict.reshape(-1)
    train_rmse = math.sqrt(mean_squared_error(trainY,train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))   

#Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')


sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
data, n = read_data(sunspots_url)
split_percent = 0.8
train_data, test_data = get_train_test(split_percent, data)
time_steps = 12
print(train_data)
print(get_train_test)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)
print('***************************************************************************')
model = create_RNN(units=3, dense_units=1, input_shape=(time_steps,1), activation=['tanh', 'tanh'])
print('***************************************************************************')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
#Make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
#Mean square error
print_error(trainY, testY, train_predict, test_predict)
plot_result(trainY, testY, train_predict, test_predict)