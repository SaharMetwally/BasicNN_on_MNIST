import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # 28*28 images size
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # load mnist dataset

'''normalizing data'''
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


'''build the model'''
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # this is the input layer for NN
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # this is the hidden layer, Dense(# of neurons, avtivation function)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # second hidden layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
validation_loss, validation_accuracy = model.evaluate(x_train, y_train)
print('validation loss = ', validation_loss)
print('validation accuracy = ', validation_accuracy)
model.save('model')
'''

'''model use'''
mew_model = tf.keras.models.load_model('model')  # load our model to use it
validation_loss, validation_accuracy = mew_model.evaluate(x_train, y_train)
print('validation loss = ', validation_loss)
print('validation accuracy = ', validation_accuracy)

predictions = mew_model.predict([x_test])
#print(predictions)
print(np.argmax(predictions[10]))
plt.imshow(x_test[10])
plt.show()
