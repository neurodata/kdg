# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf
import pickle
# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean
#%%
'''network = Sequential()
network.add(
    Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=np.shape(X_train)[1:],
    )
)
network.add(BatchNormalization())
network.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="relu",
    )
)
network.add(BatchNormalization())
network.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="relu",
    )
)
network.add(BatchNormalization())
network.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="relu",
    )
)
network.add(BatchNormalization())
network.add(
    Conv2D(
        filters=254,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="relu",
    )
)

network.add(Flatten())
network.add(BatchNormalization())
network.add(Dense(2000, activation="relu"))
network.add(BatchNormalization())
network.add(Dense(2000, activation="relu"))
network.add(BatchNormalization())
network.add(Dense(units=10, activation="softmax"))

compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 50,
        "batch_size": 64,
        "verbose": True,
        "callbacks": [callback],
    }

network.compile(**compile_kwargs)
network.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
# %%
network.save('cnn_test')'''
# %%
network = keras.models.load_model('saved_models/cifar10_ResNet164v1_model.130.h5')
print(np.mean(np.argmax(network.predict(X_test), axis=1)==y_test.reshape(-1)))
# %%
def output_from_model(model, layer_name, x):
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    output = intermediate_layer_model.predict(x)
    return output.copy()

#%%
a = output_from_model(network, 'conv2d_24', X_test[:1])
print(a.shape)
# %%
model_kdn = kdcnn(
    network=network
)
model_kdn.fit(X_train, y_train)
# %%
proba = model_kdn.predict_proba(X_test)

if np.isnan(proba).any():
    print("yes")

print(np.mean(np.argmax(proba, axis=1) == y_test.reshape(-1)))
# %%
