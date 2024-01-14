import os,sys
#os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Dropout, LayerNormalization, Activation
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip


def generate_image_patches_db(in_directory, out_directory, patch_size=64):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    for split_dir in os.listdir(in_directory):
        if not os.path.exists(os.path.join(out_directory,split_dir)):
            os.makedirs(os.path.join(out_directory,split_dir))

        for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
            if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
                os.makedirs(os.path.join(out_directory,split_dir,class_dir))
    
            for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
                im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
                patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=1)
                for i,patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
                    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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
        model.add(Dense(units=layer_size, activation=activation))
        #model.add(Activation(activation))
        model.add(Dropout(0.1))
        model.add(LayerNormalization())

    model.add(Dense(units=out_size, activation='linear', name='output'))
    model.add(Dense(units=8, activation='linear' if phase == 'test' else 'softmax'))
    
    return model

def train(block_path, model, save_path, config):

    # Define batch size
    BATCH_SIZE = 32
    epochs = 80
    initial_lr = 0.1

    dataset = keras.preprocessing.image_dataset_from_directory(
        directory=block_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(config['patch_size'], config['patch_size'])
    )

    dataset = dataset.shuffle(buffer_size=len(dataset)//10, seed=123)

    # Split dataset into train and validation sets
    train_size = int(0.85 * len(dataset))  # 85% for training
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    # Calculate class weights
    labels = [np.argmax(y.numpy()) for _, y in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    # Apply preprocessing
    preprocessing_train = keras.Sequential([
        Rescaling(1./255),
        RandomFlip("horizontal")
    ])
    
    preprocessing_validation = keras.Sequential([
        Rescaling(1./255)
    ])
    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

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
        
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)

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
    
def histogram_intersection_kernel(X, Y):
    """
    Histogram intersection kernel.
    
    Parameters:
        X: array-like of shape (n_samples_X, n_features)
        Y: array-like of shape (n_samples_Y, n_features)
    
    Returns:
        kernel_matrix: array of shape (n_samples_X, n_samples_Y)
    """
    # Expand dimensions of X and Y for broadcasting
    X_expanded = np.expand_dims(X, 1)
    Y_expanded = np.expand_dims(Y, 0)

    # Compute the minimum between each pair of vectors (broadcasting)
    minima = np.minimum(X_expanded, Y_expanded)

    # Sum over the feature dimension to compute the kernel
    kernel_matrix = np.sum(minima, axis=2)

    return kernel_matrix

def histogram_intersection_distance(X, Y):
    """
    Histogram intersection distance for kNN.
    
    Parameters:
        X: array-like of shape (n_samples_X, n_features)
        Y: array-like of shape (n_samples_Y, n_features)
    
    Returns:
        distance_matrix: array of shape (n_samples_X, n_samples_Y)
    """
    # Calculate the histogram intersection similarity
    similarity = histogram_intersection_kernel(X, Y)
    
    max_similarity = np.minimum(X.sum(axis=1)[:, np.newaxis], Y.sum(axis=1)[np.newaxis, :])
    return 1 - (similarity / max_similarity)

def accuracy(predictions, labels):
    """
    Calculates the accuracy of a set of predictions compared to the actual labels.

    Parameters:
        predictions: numpy array containing the predicted values.
        labels: numpy array containing the actual labels.

    Returns:
        A float representing the accuracy.
    """
    return sum(predictions == labels) / len(labels)

def precision(predictions, labels, class_label):
    """
    Calculates precision for a specific class in a classification task.

    Parameters:
        predictions: numpy array containing the predicted class labels.
        labels: numpy array containing the actual class labels.
        class_label: the specific class for which precision is calculated.

    Returns:
        Precision value for the specified class.
    """
    tp = np.sum((predictions == class_label) & (labels == class_label))
    fp = np.sum((predictions == class_label) & (labels != class_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(predictions, labels, class_label):
    """
    Calculates recall for a specific class in a classification task.

    Parameters:
        predictions: numpy array containing the predicted class labels.
        labels: numpy array containing the actual class labels.
        class_label: the specific class for which recall is calculated.

    Returns:
        Recall value for the specified class.
    """
    tp = np.sum((predictions == class_label) & (labels == class_label))
    fn = np.sum((predictions != class_label) & (labels == class_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def average_precision(predictions, labels):
    """
    Calculates the average precision across all classes in a classification task.

    Parameters:
        predictions: numpy array containing the predicted class labels.
        labels: numpy array containing the actual class labels.

    Returns:
        The average precision across all classes.
    """
    classes = np.unique(labels)
    return np.mean([precision(predictions, labels, c) for c in classes])

def average_recall(predictions, labels):
    """
    Calculates the average recall across all classes in a classification task.

    Parameters:
        predictions: numpy array containing the predicted class labels.
        labels: numpy array containing the actual class labels.

    Returns:
        The average recall across all classes.
    """
    classes = np.unique(labels)
    return np.mean([recall(predictions, labels, c) for c in classes])

def average_f1(predictions, labels):
    """
    Calculates the average F1 score across all classes in a classification task.

    Parameters:
        predictions: numpy array containing the predicted class labels.
        labels: numpy array containing the actual class labels.

    Returns:
        The average F1 score across all classes.
    """
    
    return 2 * average_precision(predictions, labels) * average_recall(predictions, labels) / (average_precision(predictions, labels) + average_recall(predictions, labels))

def compute_macro_roc_curve(y_onehot_test, y_score):
    """
    Computes the ROC curve and ROC area for each class.
    
    Parameters:
        y_onehot_test: array-like of shape (n_samples, n_classes)
        prob_matrix: array-like of shape (n_samples, n_classes)
    
    Returns:
        fpr_grid: Array of false positive rates at which ROC curves are evaluated.
        mean_tpr: Array of mean true positive rates corresponding to the fpr_grid.
    """
    n_classes = y_onehot_test.shape[1]
    # store the fpr, tpr
    fpr, tpr = dict(), dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    return fpr_grid, mean_tpr / n_classes

def compute_micro_roc_curve(y_onehot_test, y_score):
    """
    Computes the micro ROC curve and ROC area.
    
    Parameters:
        y_onehot_test: array-like of shape (n_samples, n_classes)
        prob_matrix: array-like of shape (n_samples, n_classes)
        
    Returns:
        fpr: Array of false positive rates.
        tpr: Array of true positive rates.
    """
    # Compute micro-average ROC curve
    fpr, tpr, _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())

    return fpr, tpr

