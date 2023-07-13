# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:09:13 2020

@author: Lenovo
"""


"""
Created on Fri Aug 14 11:42:47 2020

@author: Lenovo
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set_style('darkgrid')

datos=pd.read_csv('MSFT.csv')
nuevo=datos[['Open','High','Low','Close','Adj Close','Volume']]
g=sns.pairplot(nuevo,hue='High',diag_kind='hist')
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(),rotation=45)
datos=datos.replace(np.nan,"0")
Open=datos['Open'].values
Low=datos['Low'].values
Close=datos['Close'].values
AdjClose=datos['Adj Close'].values
Volume=datos['Volume'].values
High=datos['High'].values

X=np.array([Open,Low,Close,AdjClose,Volume]).T
Y=np.array(High)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)
error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)
print("El error es: ",error)
print("El valor de r^2 es: ",r2)
print("Los coeficientes son: \n",reg.coef_)
array=[239.570007,239.259995,243,243,27158100]
Open=array[0]
Low=array[1]
Close=array[2]
AdjClose=array[3]
Volume=array[4]
res=reg.predict([[Open,Low,Close,AdjClose,Volume]])
print("Resultados de la predicción para la variable Hight es: ",res )

