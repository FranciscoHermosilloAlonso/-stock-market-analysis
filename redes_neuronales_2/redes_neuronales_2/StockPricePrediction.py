# Importación de librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing 

# PARA LA REPRODUCIBILIDAD
np.random.seed(7)

# IMPORTANDO DATASET 
dataset = pd.read_csv('GOOG.csv', usecols=[1,2,3,4])
dataset = dataset.reindex(index = dataset.index[::-1])

# CREANDO UN ÍNDICE PARA LA FLEXIBILIDAD
obs = np.arange(1, len(dataset) + 1, 1)

# TOMAR DIFERENTES INDICADORES DE PREDICCIÓN
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

# TRAZAR TODOS LOS INDICADORES EN UN LOTE
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC promedio')
plt.plot(obs, HLC_avg, 'b', label = 'HLC promedio')
plt.plot(obs, close_val, 'g', label = 'Precio de cierre')
plt.legend(loc = 'upper right')
plt.show()

# PREPARACIÓN DE LA BASE DE DATOS DE SERIE DE TIEMPO
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# SEPARAR EL TRAIN-TEST 
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# CONJUNTO DE DATOS DE SERIE DE TIEMPO (PARA TIEMPO T, VALORES PARA TIEMPO T + 1)
trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
testX, testY = preprocessing.new_dataset(test_OHLC, 1)

# MODIFICACIÓN DE DATOS DE ENTRENAMIENTO Y PRUEBA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# MODELO LSTM
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))

# COMPILACIÓN DEL MODELO Y ENTRENAMIENTO
model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

# PREDICCIÓN
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DESNORMALIZACIÓN PARA GRAFICAR
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# ENTRENAMIENTO RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Entrenamiento RMSE: %.2f' % (trainScore))

# PRUEBA RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Prueba RMSE: %.2f' % (testScore))

# CREAR UN CONJUNTO DE DATOS SIMILAR PARA TRAZAR LAS PREDICCIONES DE ENTRENAMIENTO
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREACIÓN DE UN CONJUNTO DE DATOS SIMILAR PARA TRAZAR LAS PREDICCIONES DE PRUEBA
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

# DESNORMALIZACIÓN DEL CONJUNTO DE DATOS PRINCIPAL
OHLC_avg = scaler.inverse_transform(OHLC_avg)

# GRÁFICACIÓN DE LOS PRINCIPALES VALORES DE OHLC, PREDICCIONES DE ENTRENAMIENTO Y PREDICCIONES DE PRUEBA
plt.plot(OHLC_avg, 'g', label = 'dataset original')
plt.plot(trainPredictPlot, 'r', label = 'conjunto de entrenamiento')
plt.plot(testPredictPlot, 'b', label = 'precio de acción de la predicción / conjunto de prueba')
plt.legend(loc = 'upper right')
plt.xlabel('Tiempo en días')
plt.ylabel('Valor OHLC de las acciones')
plt.show()

# PREDECIR VALORES FUTUROS
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print("Valor del último día:", np.asscalar(last_val))
print("Valor al día siguiente:", np.asscalar(last_val*next_val))
































































































