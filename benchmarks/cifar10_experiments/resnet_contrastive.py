#%%
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
# %%
weights = []
num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_cifar100, y_cifar100), (_,_) = keras.datasets.cifar100.load_data()
y_cifar100 += 10
#x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
#y_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['y'] + 109

x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y'] + 109

x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[3],32,32,3), dtype=float)

for ii in range(x_svhn.shape[3]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp


'''x_svhn, _, y_svhn, _ = train_test_split(
                    x_svhn, y_svhn, train_size=0.1,random_state=0, stratify=y_svhn)
x_cifar100, _, y_cifar100, _ = train_test_split(
                    x_cifar100, y_cifar100, train_size=0.1,random_state=0, stratify=y_cifar100)'''

#x_ood = np.load('/cis/home/jdey4/300K_random_images.npy').astype('float32')
#y_ood = np.array(range(len(x_ood))).reshape(-1,1)+11

x_noise = np.random.random_integers(0,high=255,size=(10000,32,32,3)).astype('float')
y_noise = 120*np.ones((10000,1), dtype='float32')

x_imagenet = []
y_imagenet = []
for ii in range(10):
    #img = np.load('/Users/jayantadey/Downloads/Imagenet32_train_npz/train_data_batch_'+str(ii+1)+'.npz')
    img = np.load('/cis/home/jdey4/Imagenet32_train_npz/Imagenet32_train_npz/train_data_batch_'+str(ii+1)+'.npz')
    data = img['data']
    img_size2 = 32 * 32
    x = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))
    x_imagenet.append(x.reshape(-1,32,32,3))
    y_imagenet.append(img['labels'])

x_imagenet = np.concatenate(x_imagenet)
y_imagenet = np.concatenate(y_imagenet).reshape(-1,1) + 120

x_train = np.concatenate((x_train, x_cifar100, x_svhn, x_noise, x_imagenet))
y_train = np.concatenate((y_train, y_cifar100, y_svhn, y_noise, y_imagenet))

# Display shapes of train and test datasets
#print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
#print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# %%
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomTranslation((-.01,.01),(-.01,.01))
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
    if epoch > 450:
        lr *= 0.5e-3
    elif epoch > 350:
        lr *= 1e-3
    elif epoch > 250:
        lr *= 1e-2
    elif epoch > 150:
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
batch_size = 4056
projection_units = 512
num_epochs = 500
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

with open('pretrained_weight_contrast2.pickle', 'wb') as f:
    pickle.dump(weights, f)
# %%
sig_in = encoder_with_projection_head.predict(x_train[:20])
print(np.sqrt(np.sum((sig_in-sig_in[1])**2,axis=1)))


sig_in = encoder_with_projection_head.predict(x_test[:20])
print(np.sqrt(np.sum((sig_in-sig_in[1])**2,axis=1)))