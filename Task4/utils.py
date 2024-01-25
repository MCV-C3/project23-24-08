from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Concatenate, Input, Activation
from tensorflow.keras.models import Model

def conv_block(model, filters, kernel_size, stride, activation, use_batch_norm, d, id_layer):
    """
    Add a convolutional block to a Keras model, consisting of Conv2D, BatchNormalization (optional) and MaxPooling2D.

    Args:
    model: A Keras model instance.
    filters (int): Number of filters.
    kernel_size (int): Kernel size.
    activation (str): Activation function.
    use_batch_norm (bool): Whether to use BatchNormalization.
    d (float): Dropout rate.
    id_layer (int): Layer ID.

    Returns:
    model: A Keras model instance.
    """
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same', strides=1))
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same', strides=stride))
    if id_layer % 2 == 0:
        model.add(MaxPooling2D(pool_size=2))
    if d > 0:
        model.add(Dropout(d))
    if use_batch_norm:
        model.add(BatchNormalization())

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
    model.add(Conv2D(filters=config['filters_0'], kernel_size=3, strides=1, activation=config['activation'], padding='same', input_shape=(config['resolution'], config['resolution'], 3)))

    n_conv_blocks = config['n_conv_blocks']

    # Add convolutional blocks
    for i in range(config['n_conv_blocks']):
        model = conv_block(model, config[f'filters_{i+1}'], config[f'size_{i+1}'], config[f'stride_{i+1}'], config['activation'], config['use_batch_norm'], config['dropout'], i+1)
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

def conv_block2(x, squeeze_filters, expand_filters, activation):
    """
    Creates a convolution block.

    Args:
    x: Input tensor.
    squeeze_filters (int): Number of filters in the squeeze layer.
    expand_filters (int): Number of filters in the expand layer.

    Returns:
    x: Output tensor.
    """
    # Squeeze layer
    squeeze = Conv2D(squeeze_filters, (1, 1), padding='same', activation=activation)(x)
    
    # Expand layer (with a mix of 1x1 and 3x3 convolutions)
    expand_1x1 = Conv2D(expand_filters, (1, 1), padding='same', activation=activation)(squeeze)
    expand_3x3 = Conv2D(expand_filters, (3, 3), padding='same', activation=activation)(squeeze)
    
    # Concatenate 1x1 and 3x3 expand outputs
    x = Concatenate()([expand_1x1, expand_3x3])

    return x

def build_model2(config):
    """

    Args:

    Returns:
    model: A Keras model instance.
    """
    inputs = Input((config['resolution'], config['resolution'], 3))

    # Initial convolution layer
    x = Conv2D(config['filters_0'], kernel_size=1, strides=1, padding='same', activation=config['activation'])(inputs)
    x = MaxPooling2D(pool_size=2)(x)

    for i in range(config['n_conv_blocks']):
        x = conv_block2(x, squeeze_filters=config[f'filters_{i+1}'], expand_filters=config[f'filters_{i+1}'], activation=config['activation'])
        x = conv_block2(x, squeeze_filters=config[f'filters_{i+1}'], expand_filters=config[f'filters_{i+1}'], activation=config['activation'])
        if (i+1) % 2 == 0:
            x = MaxPooling2D(pool_size=2)(x)

    # Final layers
    x = Conv2D(8, (1, 1), padding='same', activation=config['activation2'])(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model

