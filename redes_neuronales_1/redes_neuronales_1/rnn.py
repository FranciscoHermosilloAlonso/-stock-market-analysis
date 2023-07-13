# Recurrent Neural Network

# Parte 1 - Preprocesamiento de datos

# Importación las bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el conjunto de entrenamiento
# Solo se predice el "precio de las acciones abiertas" para la empresa extrayendo una columna.
training_set = pd.read_csv('GOOG_Train.csv')
# Obteniendo solo el precio de las acciones abiertas para la entrada de RNN.
# Para convertir la forma vectorial de una sola columna en una forma de matriz, usaremos 1: 2 como índice de columna. 
# La segunda columna será ignorada y obtendremos nuestra Columna de precio de acciones abiertas en forma de matriz.
# La salida será una matriz 2d Numpy.
training_set = training_set.iloc[:,1:2].values

# Escala de funciones
# Utilizará la normalización como función de escala.
# El rango predeterminado para MinMaxScaler es de 0 a 1, que es lo que queremos. Así que no hay argumentos en él.
# Se ajustará al juego de entrenamiento y lo escalará y reemplazará el juego original.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Obteniendo las entradas y salidas
# Restringir la entrada y la salida según el funcionamiento de LSTM.
X_train = training_set[0:4121]
y_train = training_set[1:4122]

# Remodelación: adición de intervalo de tiempo como dimensión de entrada.
X_train = np.reshape(X_train, (4121, 1, 1))

# Parte 2 - Construyendo la RNN

# Importación de las bibliotecas y paquetes de Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Inicializando el RNN
# Creando un objeto de clase Sequential para crear el RNN.
regressor = Sequential()

# Agregar la capa de entrada y la capa LSTM
# 4 unidades de memoria, función de activación sigmoidea y (ningún intervalo de tiempo con 1 atributo como entrada)
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Añadiendo la capa de salida
# 1 neurona en la capa de salida para salida unidimensional
regressor.add(Dense(units = 1))

# Compilando el RNN
# Compilando todas las capas juntas.
# La pérdida ayuda en la manipulación de pesos en NN.
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Adaptación del RNN al equipo de entrenamiento
# Número de épocas aumentado para una mejor convergencia.
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Parte 3 - Hacer las predicciones y visualizar los resultados

# Obtener el precio real de las acciones de 2021
test_set = pd.read_csv('GOOG_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Obtener el precio de las acciones previsto para 2021
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizando los resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio real de las acciones')
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio de acción predecido')
plt.title('Predicción del precio de las acciones')
plt.xlabel('Tiempo')
plt.ylabel('Precio de mercado')
plt.legend()
plt.show()


# Hacer predicciones para todo el conjunto de datos
# Obtención del precio real de las acciones de 2012 a 2016
real_stock_price_train = pd.read_csv('GOOG_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

# Obtención del precio de las acciones previsto de 2004 a 2020
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualizando los resultados
plt.plot(real_stock_price_train, color = 'red', label = 'Precio real de las acciones')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Precio de acción previsto')
plt.title('Predicción del precio de las acciones')
plt.xlabel('Tiempo')
plt.ylabel('Precio de mercado')
plt.legend()
plt.show()


# Parte 4 - Evaluación de la RNN

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('precisión:', rmse)