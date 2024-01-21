# C3. Image Classification

## Folder structure 
The code and data is structured as follows:

        .
        ├── train.py                        # Source code
        ├── Explainability.ipynb            # Explainability of the results
        └── utils.py                        



## Requirements
Standard Computer Vision python packages are used. Regarding the python version, Python >= 3.9 is needed.

- OpenCV / cv2: $pip install opencv-python
- Tqdm: $pip installl tqdm
- Pickle: $pip install pickle5
- Sklearn: $pip install -U scikit-learn
- TensorFlow: $pip install tensorflow


## Usage
Some code is in .ipynb format, thus it is required to have Jupyter Notebook or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the Jupyter extension.

To replicate the submitted results and ensure that everything works as expected, simply use the Run all button (or run the code blocks from top to bottom).

## Tasks
This lab can be divided into two main objectives: 
1. Compare the performance of the model being fine-tuned using the whole dataset or only a fraction of it.
2. Modify the model by adding layers at the end or unfreezing layers and retraining them, or both. While also optimizing other hyperparameters.

The aim is to compare both tasks to see if there are any performance differences and extract some conclusions from it. Moreover, to understand how the model behaves at a more general level.