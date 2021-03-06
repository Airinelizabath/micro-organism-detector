import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TEST_DIR = 'C:\\Users\\user\\Downloads\\project\\test'
IMG_SIZE = 100
LR = 1e-3
MODEL_NAME = 'maleria'

def create_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split('.')[0][0]
		img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		#img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img_data), img_num])
	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data
	 
test_data = create_test_data()
 
def create_model():
    tf.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu', name='d1')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)
    return model

model=create_model()
#model load code
model.load('C:\\Users\\user\\Downloads\\project\\model.tfl')
dense1_vars = tflearn.variables.get_layer_variables_by_name('d1')
print("Dense1 layer weights:")
print(model.get_weights(dense1_vars[0]))

#model new input kodkande code . code below does sample data test. can be left unchanged.

fig = plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):

	img_num = data[1]
	img_data = data[0]
	

	y = fig.add_subplot(4, 4, num + 1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	model_out = model.predict([data]) [0]

	if np.argmax(model_out) == 1:
		str_label = 'Not Infected'+img_num
	else:
		str_label = 'Infected'+img_num

	y.imshow(orig, cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()
