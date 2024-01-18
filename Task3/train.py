import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import wandb

def add_custom_layers(x, num_layers, activation, dropout, use_batch_norm):
    """
    Adds customizable layers to the base model output.

    :param x: Output tensor from the base model.
    :param num_layers: Number of layers to add.
    :param activation: Activation functions for all layer.
    :param dropout: Wether to use the dropout regularization.
    :param use_batch_norm: Wether to use batch normalization.
    :return: Output tensor after adding the custom layers.
    """
    if num_layers == 1:
        neurons = [512]
    elif num_layers == 2:
        neurons = [512, 256]
    elif num_layers == 3:
        neurons = [512, 256, 128]
    
    for i in range(num_layers):
        x = Dense(neurons[i], activation=activation)(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    
    return x


def train():

    wandb.init()
    # Get hyperparameters
    config = wandb.config

    # Define constants
    IMG_WIDTH, IMG_HEIGHT = config['resolution'], config['resolution']
    MODEL_PATH = './pretrained/model.h5'
    DATASET_DIR = './MIT_split'

    # Define the data generator for data augmentation and preprocessing
    train_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        vertical_flip=False,
        validation_split=0.2  # Set the validation split (only for MIT_split dataset)
    )

    # Load and preprocess the training and validation datasets
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
        subset='training'  # Specify this is training data (only for MIT_split dataset)
    )

    val_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
        subset='validation' # Specify this is validation data (only for MIT_split dataset)
    )

    # Define the data generator for preprocessing (no augmentation for test data)
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load and preprocess the test dataset
    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False  # No need to shuffle the test data
    )
    
    # Load EfficientNetB0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Unfreeze the last N layers
    N = config['n_layers_unfreeze']  # Number of layers to unfreeze
    for layer in base_model.layers[:-N]:
        layer.trainable = False
    for layer in base_model.layers[-N:]:
        layer.trainable = True

    # Add classifier layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=config['activation'])(x)
    if config['num_layers'] > 0:
        x = add_custom_layers(x, config['num_layers'], config['activation'], config['dropout'], config['use_batch_norm'])
    predictions = Dense(8, activation='softmax')(x)  # Assuming 8 classes

    model = Model(inputs=base_model.input, outputs=predictions)
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # for layer in model.layers:
    #     if layer.trainable == True:
    #         print(layer.name)

    final_learning_rate = 1e-5
    learning_rate_decay_factor = (final_learning_rate / config['lr'])**(1/config['epochs'])
    steps_per_epoch = int(train_dataset.samples/config['batch_size'])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['lr'],
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True)
    
    es_cback = EarlyStopping(monitor='val_accuracy', mode='max', patience=7, min_delta=0.0001)
    # checkpoint_cback = ModelCheckpoint(filepath=MODEL_PATH, mode='max', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    cbacks = [es_cback]#, checkpoint_cback]

    if config['optimizer_type'] == 'adam':
        optimizer = Adam(learning_rate=lr_schedule, weight_decay=config['l2'])
    elif config['optimizer_type'] == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=config['momentum'], weight_decay=config['l2'])

    auc = tf.keras.metrics.AUC(num_thresholds=200, name='PR-AUC', curve='PR')

    # Compile the model
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', auc])

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=cbacks
    )

    for epoch in range(len(history.history['loss'])):
        wandb.log({
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
        })

    # Load the trained model
    # model.load_weights(MODEL_PATH)

    # Evaluate the model on the test data
    # loss, acc, auc = model.evaluate(test_dataset)

    # print(f"Test Loss: {loss}")
    # print(f"Test Accuracy: {acc}")
    # print(f"Test AUC: {auc}")
    wandb.finish()


sweep = True
if sweep:
    sweep_id = "c3-mcv/cnn/0yk44dh5"
    wandb.agent(sweep_id, train, count=2)
else:
    config = {
        'lr': 1e-3,
        'batch_size': 64,
        'epochs': 2,
        'activation': 'relu',
        'optimizer_type': 'adam',
        'momentum': 0.9,
        'n_layers_unfreeze': 10,
        'num_layers': 1,
        'dropout': 0.1,
        'use_batch_norm': True,
        'l2': 0.001,
        'resolution': 224

    }

    # Initialize wandb with a sample configuration
    wandb.init(project='cnn', entity='c3-mcv', config=config)

    # Train the model
    train()