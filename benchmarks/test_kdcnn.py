# %%
from bitarray import bitarray
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as bknd
import pickle
from kdg import kdcnn, kdf, kdn
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
'''a = output_from_model(network, 'conv2d_24', X_test[:1])
print(a.shape)'''
# %%
model_kdn = kdn(
    network=network,
    verbose=False,
)
model_kdn.fit(X_train[:6000], y_train[:6000])
# %%
proba = model_kdn.predict_proba(X_test)

'''if np.isnan(proba).any():
    print("yes")'''

print(np.mean(np.argmax(proba, axis=1) == y_test.reshape(-1)))
# %%
def _get_polytope_ids(X):
    total_samples = X.shape[0]
        
    outputs = [] 
    inp = model_kdn.network.input

    for layer in model_kdn.network.layers:
        if 'activation' in layer.name:
            outputs.append(layer.output) 

    functor = bknd.function(inp, outputs)
    layer_outs = functor(X)

    activation = []
    for layer_out in layer_outs:
        #print(layer_out.shape)
        #print((layer_out>0).astype('int').reshape(total_samples, -1))
        activation.append(
            (layer_out>0).astype('bool').reshape(total_samples, -1)
        )
    polytope_ids = np.concatenate(activation, axis=1)

    return polytope_ids
# %%
'''polytopes = np.unique(
        polytope_ids, axis=0
        )
for polytope in polytopes:
    #indx = np.where(polytope==0)[0]
    polytope_ = polytope.copy()

    matched_pattern = (polytope_ids==polytope_)
    matched_nodes = np.zeros((len(polytope_ids),model_kdn.total_layers))
    end_node = 0
    normalizing_factor = 0
    for layer in range(model_kdn.total_layers):
        end_node += model_kdn.network_shape[layer]
        matched_nodes[:, layer] = \
            np.sum(matched_pattern[:,end_node-model_kdn.network_shape[layer]:end_node], axis=1)\
                + 1/model_kdn.network_shape[layer]

        normalizing_factor += \
            np.log(np.max(matched_nodes[:, layer]))

    scales = np.exp(np.sum(np.log(matched_nodes), axis=1)\
        - normalizing_factor)
    print(scales)'''
# %%
