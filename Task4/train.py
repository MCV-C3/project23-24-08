import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import wandb
from utils import build_model


def train():

    wandb.init()
    # Get hyperparameters
    config = wandb.config

    # Define constants
    IMG_WIDTH, IMG_HEIGHT = config['resolution'], config['resolution']
    MODEL_PATH = './pretrained/model.h5'
    DATASET_DIR = '../MIT_small_train_1'

    # Define the data generator for data augmentation and preprocessing
    train_data_generator = ImageDataGenerator(
        preprocessing_function=lambda x: x/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        vertical_flip=False,
    )

    # Load and preprocess the training and validation datasets
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
    )

    val_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
    )

    # Define the data generator for preprocessing (no augmentation for test data)
    test_data_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255.0)

    # Load and preprocess the test dataset
    test_dataset = test_data_generator.flow_from_directory(
        directory='../MIT_split/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False  # No need to shuffle the test data
    )
    
    model = build_model(config)
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    final_learning_rate = 1e-5
    learning_rate_decay_factor = (final_learning_rate / config['lr'])**(1/config['epochs'])
    steps_per_epoch = int(train_dataset.samples/config['batch_size'])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['lr'],
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True)
    
    es_cback = EarlyStopping(monitor='val_accuracy', mode='max', patience=7, min_delta=0.0001)
    checkpoint_cback = ModelCheckpoint(filepath=MODEL_PATH, mode='max', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    cbacks = [es_cback, checkpoint_cback]

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
    model.load_weights(MODEL_PATH)

    # Evaluate the model on the test data
    loss, acc, auc = model.evaluate(test_dataset)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {acc}")
    print(f"Test AUC: {auc}")
    wandb.finish()


sweep = False
if sweep:
    sweep_id = "c3-mcv/cnn2/"
    wandb.agent(sweep_id, train, count=2)
else:
    config = {
        'lr': 1e-3,
        'batch_size': 64,
        'epochs': 5,
        'activation': 'relu',
        'activation2': 'relu',
        'optimizer_type': 'adam',
        'momentum': 0.9,
        'dropout': 0.1,
        'use_batch_norm': True,
        'l2': 0.001,
        'resolution': 256,
        'n_conv_blocks': 2,
        'n_dense_layers': 2,
        'filters_0': 8,
        'size_0': 3,
        'pool_size_0': 2,
        'filters_1': 16,
        'size_1': 3,
        'pool_size_1': 2,
        'filters_2': 32,
        'size_2': 3,
        'pool_size_2': 2,
        # 'filters_3': 64,
        # 'size_3': 3,
        # 'pool_size_3': 2,
        # 'filters_4': 128,
        # 'size_4': 3,
        # 'pool_size_4': 2,
        # 'filters_5': 256,
        # 'size_5': 3,
        # 'pool_size_5': 2,
        # 'filters_6': 512,
        # 'size_6': 3,
        # 'pool_size_6': 2,
    }

    # Initialize wandb with a sample configuration
    wandb.init(project='cnn2', entity='c3-mcv', config=config)

    # Train the model
    train()