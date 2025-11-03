
#Experiment 5

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


#data
X = np.linspace(0,1,1000)
y=5*X+7+np.random.randn(1000)

#build the architecture
model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

#compile
model.compile(optimizer='adam',loss='mean_squared_error')
 
#Build a model
res=model.fit(X,y,epochs=50)


#predict
predict=model.predict(X)


#visualization
plt.scatter(X,y,label='original data',color='blue')

plt.plot(X,predict,label='Predictions',color='red')
plt.title("Optimizer: Adam")
plt.legend()
#plt.plot(res.history['loss'])
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
