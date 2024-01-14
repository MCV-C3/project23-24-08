#from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Dropout, LayerNormalization

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@tf.autograph.experimental.do_not_convert
def map_preprocess_train(x, y):
    preprocessing_train = keras.Sequential([
        keras.layers.Rescaling(1./255),
        keras.layers.RandomFlip("horizontal")
    ])
    return preprocessing_train(x, training=True), y

@tf.autograph.experimental.do_not_convert
def map_preprocess_validation(x, y):
    preprocessing_validation = keras.Sequential([
        keras.layers.Rescaling(1./255)
    ])
    return preprocessing_validation(x, training=False), y

def build_mlp(in_size, out_size, num_layers, activation, phase='train'):
    model = Sequential()
    model.add(Input(shape=(in_size, in_size, 3), name='input'))
    model.add(Reshape((in_size*in_size*3,)))

    if in_size*in_size*3 < out_size: 
        #increment size
        size_step = (out_size - in_size*in_size*3) / num_layers
        sign = 1
    else: 
        #decrement size 
        size_step = (in_size*in_size*3 - out_size) / num_layers
        sign = -1

    # Add layers
    for i in range(num_layers - 1):
        layer_size = int(in_size*in_size*3 + sign * size_step * i)
        model.add(Dense(units=layer_size, activation=activation, name = 'dense_'+str(i)))
        model.add(Dropout(0.1))
        model.add(LayerNormalization())

    model.add(Dense(units=out_size, activation=activation, name='output'))
    model.add(Dense(units=8, activation='linear' if phase == 'test' else 'softmax'))
    
    return model

def train(block_path, model, save_path, config):

    # Define batch size
    BATCH_SIZE = 64
    epochs = 10
    initial_lr = 0.1

    dataset = keras.utils.image_dataset_from_directory(
        directory=block_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(config['patch_size'], config['patch_size'])
    )

    dataset = dataset.shuffle(buffer_size=len(dataset)//2, seed=123)

    # Split dataset into train and validation sets
    train_size = int(0.85 * len(dataset))  # 85% for training
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    # Calculate class weights
    labels = [np.argmax(y.numpy()) for _, y in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    # Apply preprocessing
    train_dataset = train_dataset.map(map_preprocess_train)
    validation_dataset = validation_dataset.map(map_preprocess_validation)

    # Prefetch
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    auc = tf.keras.metrics.AUC(num_thresholds=200, name='PR-AUC', curve='PR')

    final_learning_rate = 0.0001
    learning_rate_decay_factor = (final_learning_rate / initial_lr)**(1/epochs)
    steps_per_epoch = int(train_size/BATCH_SIZE)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics=['accuracy', auc])

    # Train your model with class weights
    model.fit(train_dataset,
              epochs=epochs,
              validation_data=validation_dataset,
              verbose=2,
              class_weight=class_weights,
              callbacks=[callback])
    

    print(model.summary())
    print('Saving the model into '+save_path+' \n')
    model.save_weights(save_path)
    print('Done!\n')
