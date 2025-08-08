#To implement MLP on MNIST dataset using keras
import numpy as np 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocess the data
y_train = to_categorical(y_train) 
y_test1 = to_categorical(y_test)

#build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input
model.add(Dense(units=10, activation='relu'))  # Hidden layer with 256 neurons
model.add(Dense(units=128, activation='relu'))  # Hidden layer with 128 neurons
model.add(Dense(units=10,activation='softmax'))

# Compile the model
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation split
result = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test1)) 
print(result.history.keys())
print(result.history.items())

#evaluate the model
loss, accuracy = model.evaluate(x_test, y_test1)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

#predict
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1) 
print(f'Predicted class for first test sample: {predicted_classes[1]}')

#print the predicted image
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.title(f'actual :{y_test[i]} predicted :{predicted_classes[i]}')
    plt.imshow(x_test[i], cmap='gray')
plt.show()

#visualization of the model
plt.plot(result.history['loss'],label='train loss',color='blue')
plt.plot(result.history['val_loss'],label='val loss',color='red')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#visualization of the accuracy
plt.plot(result.history['accuracy'],label='train accuracy',color='blue')
plt.plot(result.history['val_accuracy'],label='val accuracy',color='red')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
