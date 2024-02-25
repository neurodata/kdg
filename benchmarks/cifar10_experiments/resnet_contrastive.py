#%%
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from tensorflow.keras.callbacks import LearningRateScheduler
# %%
weights = []
num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
#x_ood = np.load('/cis/home/jdey4/300K_random_images.npy').astype('float32')
#y_ood = np.array(range(len(x_ood))).reshape(-1,1)+11

x_noise = np.random.random_integers(0,high=255,size=(10000,32,32,3)).astype('float')
y_noise = 10*np.ones((10000,1), dtype='float32')

x_train = np.concatenate((x_train, x_noise))
y_train = np.concatenate((y_train, y_noise))

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# %%
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomRotation(0.02),
        layers.RandomContrast(.02),
        layers.RandomBrightness(factor=0.02)
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_train)

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
    lr = 1e-3
    if epoch > 900:
        lr *= 0.5e-3
    elif epoch > 700:
        lr *= 1e-3
    elif epoch > 500:
        lr *= 1e-2
    elif epoch > 300:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]
#%%
def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


encoder = create_encoder()
encoder.summary()

learning_rate = 0.001
batch_size = 1024
hidden_units = 512
projection_units = 128
num_epochs = 1000
dropout_rate = 0.5
temperature = 0.05
# %%
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model
# %%
encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit(
    x=x_train, y=y_train, batch_size=batch_size, 
    epochs=num_epochs,
    callbacks=callbacks
)

#%%
for layer_id, layer in enumerate(encoder_with_projection_head.layers):
    pretrained_weights = encoder_with_projection_head.layers[layer_id].get_weights()
    weights.append(
        pretrained_weights
    )

with open('pretrained_weight_contrast.pickle', 'wb') as f:
    pickle.dump(weights, f)
# %%
sig_in = encoder_with_projection_head.predict(x_train[:20])
print(np.sqrt(np.sum((sig_in-sig_in[1])**2,axis=1)))


sig_in = encoder_with_projection_head.predict(x_test[:20])
print(np.sqrt(np.sum((sig_in-sig_in[1])**2,axis=1)))