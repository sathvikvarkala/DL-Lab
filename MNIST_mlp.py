#To implement MLP on MNIST dataset using Keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist 
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt


#Load Data
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Prepocessing
y_train=to_categorical(y_train)
y_test1=to_categorical(y_test)


#Build the architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

#Compile
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


#Train
result=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test1))
print(result.history.keys())
print(result.history.items())

#Evaluate
loss,accuracy=model.evaluate(x_test,y_test1)
print(f"test loss:{loss},\ntest_accuracy:{accuracy}")

#Predict
predictions=model.predict(x_test)
predict_label=np.argmax(predictions,axis=1)
print(f"predicted label:{predict_label[1]}")

#Print the predicted image
for i in range(5):
    #print(f"actual:{y_test[i]}\npredicted:{predict_label[i]}")
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"actual:{y_test[i]}\npredicted:{predict_label[i]}")
plt.show()
#Visualization
plt.plot(result.history['loss'],label='train loss',color='blue')
plt.plot(result.history['val_loss'],label='valdation loss',color='red')
plt.xticks(np.arange(1,50))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()

plt.plot(result.history['accuracy'],label='train accuracy',color='blue')
plt.plot(result.history['val_accuracy'],label='valdation accuracy',color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()

