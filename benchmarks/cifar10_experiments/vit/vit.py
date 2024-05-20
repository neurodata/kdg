#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, Activation, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.datasets import cifar10
from vit_keras import vit, utils
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import gc
import numpy as np
# %%
seeds = [3]#2022

# %%
(train_data, train_label), (test_data, test_label) = cifar10.load_data()
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
train_data = (train_data/255.).astype("float16")
test_data = (test_data/255.).astype("float16")
# %%
for seed in seeds:
    print('Doing seed ', seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_label, random_state=seed, shuffle=True)
    ######
    batch_size = 32
    datagen = ImageDataGenerator()
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    ######
    input_shape = (32, 32, 3) #Cifar10 image size
    image_size = 256 #size after resizing image
    num_classes = 10

    def build_model():
        inputs = Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (image_size, image_size)))(inputs) #Resize image to  size 224x224
        base_model = vit.vit_b16(image_size=image_size, activation="sigmoid", pretrained=True,
                                include_top=False, pretrained_top=False)
        
        base_model.trainable = False #Set false for transfer learning
        x = base_model(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model_final = Model(inputs=inputs, outputs=outputs)
        return model_final

    ##################
    model = build_model()
    '''model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print("\n")
    model.fit(X_train,
            y_train,
            batch_size=batch_size,
            steps_per_epoch=200,
            epochs=2,
            validation_data=(X_valid, y_valid),
            )
    gc.collect()'''

    ############
    #Set training callbacks
    plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=1, verbose=1)
    earlystopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    #Switch ViT layer to trainable for fine tuning
    for layer in model.layers:
        layer.trainable = True
        
    #Requires compile again to activate trainable=True
    model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print("\n")
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        steps_per_epoch=200, #you can delete this parameter to achieve more accuracy, but takes much time.
                        epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[plateau, earlystopping]
                    )
    print("\nTest Accuracy: ", accuracy_score(np.argmax(test_label, axis=1), np.argmax(model.predict(test_data), axis=1)))

    model.save('vit_model_'+str(seed)+'.keras')
# %%
