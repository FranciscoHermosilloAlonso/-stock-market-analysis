import numpy as np 


# FUNCIÓN PARA CREAR DATOS 1D EN EL CONJUNTO DE DATOS DE SERIE DE TIEMPO
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# ESTA FUNCIÓN PUEDE UTILIZARSE PARA CREAR UN CONJUNTO DE DATOS DE SERIE DE TIEMPO A PARTIR DE CUALQUIER ARREGLO 1D	