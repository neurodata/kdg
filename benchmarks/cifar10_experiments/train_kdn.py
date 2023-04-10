# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn
import pickle
from tensorflow.keras.datasets import cifar10
#%%
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

tasks = [[0,1], [2,3], [4,5],
         [6,7], [8,9]]

#%%
for task, labels in enumerate(tasks):
    if task==0 or task==1 or task==2:
        continue
    
    print("Doing task ", task)

    nn_file = 'resnet_models/cifar_model_'+str(task)
    network = keras.models.load_model(nn_file)
    # Convert class vectors to binary class matrices.
    idx_to_train = []
    idx_to_test = []

    for label in labels:
        idx_to_train.extend(np.where(y_train==label)[0])
        idx_to_test.extend(np.where(y_test==label)[0])

    _, y_train_task = np.unique(y_train[idx_to_train], return_inverse=True)
    _, y_test_task = np.unique(y_test[idx_to_test], return_inverse=True)

    model_kdn = kdcnn(
        network=network
    )
    model_kdn.fit(x_train[idx_to_train], y_train_task, batch=10)
    predicted_label = model_kdn.predict(x_test[idx_to_test])

    print('Accuracy task ', task, ' ', np.mean(predicted_label==y_test_task))
    with open('kdn_models/kdn_'+str(task)+'.pickle', 'wb') as f:
        pickle.dump(model_kdn, f)