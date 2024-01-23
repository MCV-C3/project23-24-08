from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def conv_block(model, filters, kernel_size, pool_size, activation, use_batch_norm):
    """
    Add a convolutional block to a Keras model, consisting of Conv2D, BatchNormalization (optional) and MaxPooling2D.

    Args:
    model: A Keras model instance.
    filters (int): Number of filters.
    kernel_size (int): Kernel size.
    activation (str): Activation function.
    use_batch_norm (bool): Whether to use BatchNormalization.

    Returns:
    model: A Keras model instance.
    """
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))

    return model

def build_model(config):
    """
    Build a customizable CNN model with Global Average Pooling.

    Args:
    config (dict): A dictionary containing model configuration parameters.

    Returns:
    model: A Keras model instance.
    """

    model = Sequential()

    # First layer with input shape specified
    model.add(Conv2D(filters=config['filters_0'], kernel_size=config['size_0'], activation=config['activation'], padding='same', input_shape=(config['resolution'], config['resolution'], 3)))
    model.add(MaxPooling2D(pool_size=config['pool_size_0']))

    n_conv_blocks = config['n_conv_blocks']

    # Add convolutional blocks
    for i in range(config['n_conv_blocks']):
        model = conv_block(model, config[f'filters_{i+1}'], config[f'size_{i+1}'], config[f'pool_size_{i+1}'], config['activation'], config['use_batch_norm'])
    # Global Average Pooling layer
    model.add(GlobalAveragePooling2D())

    neurons = config[f'filters_{n_conv_blocks}']

    # Add dense layers
    for i in range(config['n_dense_layers']):
        model.add(Dense(neurons, activation=config['activation2']))
        if config['dropout'] > 0:
            model.add(Dropout(config['dropout']))
        if config['use_batch_norm']:
            model.add(BatchNormalization())
        neurons //= 2

    # Output layer
    model.add(Dense(8, activation='softmax'))

    return model
