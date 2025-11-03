#Experiment 1
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
model.compile(optimizer='sgd',loss='mean_squared_error')
 
#Build a model
model.fit(X,y,epochs=1)




#visualization
plt.scatter(X,y,label='original data')
plt.title('Optimizer: SGD, Epoch:1')
plt.show()
