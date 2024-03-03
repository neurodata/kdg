#%%
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
#%%
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 16:
        lr *= 1e-3
    elif epoch > 8:
        lr *= 1e-2
    elif epoch > 4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
# %%
input_shape = (32,32,3)
num_classes = 10
seed = 0
batch_size = 32
num_epochs = 20

model = keras.Sequential()
base_model = ResNet50V2(
        include_top=True, weights='imagenet', pooling="avg",
        classifier_activation=None
    )

inputs = keras.Input(shape=input_shape)
model.add(inputs)
model.add(UpSampling2D(size=(7,7)))
model.add(base_model)
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(
            Dense(
                    num_classes,
                    activation='softmax'
                )
        )

model.build()
# %%
model.layers[1].trainable = False
#%%
model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=['accuracy'])


#%%
filepath = 'cifar10.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=5,
                            min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
#%%
# Load data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
# %%
x_train, x_cal, y_train, y_cal = train_test_split(
                x_train, y_train, train_size=0.9, random_state=seed, stratify=y_train)

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_cal_one_hot = keras.utils.to_categorical(y_cal, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
# %%
model.fit(x_train, y_train_one_hot,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test_one_hot),
        shuffle=True,
        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test_one_hot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# %%
model.save('resnet20_models/cifar_finetune10_'+str(seed))
# %%
