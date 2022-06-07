# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn
import pickle
# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
#%%

# %%
network = keras.models.load_model('cnn_test')
np.mean(np.argmax(network.predict(X_test), axis=1)==y_test.reshape(-1))
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
    network=network,
    k=1e300,
    verbose=False,
)
model_kdn.fit(X_train, y_train)

# %%
print(np.mean(model_kdn.predict(X_test) == y_test.reshape(-1)))
# %%
