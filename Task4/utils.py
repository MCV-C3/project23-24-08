from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

def squeeze_block(x, squeeze_planes, expand1x1_planes, expand3x3_planes, activation):
    squeeze = layers.Conv2D(squeeze_planes, kernel_size=1, activation=activation)(x)

    expand1x1 = layers.Conv2D(expand1x1_planes, kernel_size=1, activation=activation)(squeeze)
    expand3x3 = layers.Conv2D(expand3x3_planes, kernel_size=3, padding='same', activation=activation)(squeeze)

    return tf.concat([expand1x1, expand3x3], axis=3)

def build_model(config, num_classes):
    inputs = layers.Input(shape=(config['resolution'], config['resolution'], 3)) 

    x = layers.Conv2D(config['filters_0'], kernel_size=3, strides=1, padding='same', activation=config['activation'])(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    if config['dropout'] > 0:
        x = layers.Dropout(config['dropout'])(x)
    if config['use_batch_norm']:
        x = layers.BatchNormalization()(x)

    for i in range(config['n_conv_blocks']):
        x = squeeze_block(x, config[f'filters_sq_{i+1}'], config[f'filters_ex_{i+1}'], config[f'filters_ex_{i+1}'], config['activation'])
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        
        if config['dropout'] > 0:
            x = layers.Dropout(config['dropout'])(x)
        if config['use_batch_norm']:
            x = layers.BatchNormalization()(x)

    x = layers.Conv2D(num_classes, kernel_size=3, strides=1, padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    return Model(inputs=inputs, outputs=x)
    
def preprocess(image, label):
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0
    return image, label

def get_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(preprocess).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(preprocess).batch(batch_size)

    return train_dataset, test_dataset
     