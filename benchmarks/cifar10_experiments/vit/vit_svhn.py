#%%
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, Activation, Input, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle
#from tensorflow.keras.layers.core import Lambda
from vit_keras import vit, utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
# %%
seeds = [0,1,2,3]#2022

# %%
data = loadmat('/work/wyw112/SVHN/train_32x32.mat')
trainset, train_label = data['X'], data['y']

train_label = to_categorical(train_label)
#cross_label = to_categorical(cross_label)
trainset = (trainset/255.).astype("float16")
#crossdataset = (crossdataset/255.).astype("float16")
# %%
for seed in seeds:
    print('Doing seed ', seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_valid, y_train, y_valid = train_test_split(trainset, train_label, random_state=seed, shuffle=True)
    ######
    batch_size = 32
    datagen = ImageDataGenerator()
    #Generate batches of augmented images
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
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    model.summary()
    print("\n")

    best_accuracy = 0.0  # Initialize best accuracy
    # best_model_path = '/best_model/best_model.h5'  # Path to save the best model



    for epoch in range(10):
        model.fit(train_generator, steps_per_epoch=200, epochs=1)
        gc.collect()  #garbage collector to free up memory

        # Evaluate model on validation data
        #validation_accuracy = model.evaluate(X_valid, y_valid, verbose=0)[1]

        # if validation_accuracy > best_accuracy:
        #     best_accuracy = validation_accuracy
        #     model.save(best_model_path)  # Save the best model checkpoint

        #print(f"Epoch {epoch + 1} - Validation Accuracy: {validation_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

    print("Training complete.")

    # with open('vit_model_cifar100_'+str(seed)+'.pickle', 'wb') as f:
    #     pickle.dump(model, f)
    model.save('vit_model_svhn_'+str(seed)+'.keras')
# %%
