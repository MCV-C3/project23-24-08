# C3. Image Classification

## Folder structure 
The code and data is structured as follows:

        .
        ├── code.ipynb                          # Source code
        ├── mlp_MIT_8_scene.py                  # Initial provided code
        └── patch_based_mlp_MIT_8_scene.py      # Datasets



## Requirements
Standard Computer Vision python packages are used. Regarding the python version, Python >= 3.9 is needed.

- OpenCV / cv2: $pip install opencv-python
- Tqdm: $pip installl tqdm
- Pickle: $pip install pickle5
- Sklearn: $pip install -U scikit-learn


## Usage
The source code is in .ipynb format, thus it is required to have Jupyter Notebook or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the Jupyter extension.

To replicate the submitted results and ensure that everything works as expected, simply use the Run all button (or run the code blocks from top to bottom).

## Tasks
The main goal of this week's project is to develop an image classification system, and to compare the performance of using handcrafted techniques (DenseSIFT) with simple learning tecnhiques (MLP). We conducted several tasks to optimize and test the classification system (while also explaining all results found), such as:

1. MLP as a classifier

    1.1 Add/change layers in the network topology
   
    1.2 Given an image, get the output of a given layer and apply SVM on it.
   
    1.3 Given an image, get the output using the MLP as classifier.
   
    1.4 Compare best performance from 1.3 and 1.4

4. MLP as a dense descriptor

    2.1 Divide the image into small patches, extract features of each of them with the MLP and use them as features for the BoVW.
   
    2.2 Divide the image into small patches, extract features of each of them with Dense SIFT and use them as features for the BoVW.

    2.3 Compare best performance from 2.1 and 2.2



All the hyperparameters are optimized using wandb.ai.
