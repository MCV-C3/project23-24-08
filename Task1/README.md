# C3. Image Classification

## Folder structure 
The code and data is structured as follows:

        .
        ├── BagofVisualWords.ipynb          # Source code
        ├── Explainability.ipynb            # Explainability of the results
        └── MIT_split                       # Datasets
            ├── test                        # Test sets
            │   ├── coast
            │   ├── forest
            │   ├── highway
            │   ├── inside_city
            │   ├── mountain
            │   ├── Opencountry
            │   ├── street
            │   └── tallbuilding
            └── train                       # Train sets
                ├── coast
                ├── forest
                ├── highway
                ├── inside_city
                ├── mountain
                ├── Opencountry
                ├── street
                └── tallbuilding


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
The main goal of this project is to develop an image classification system with handcrafted techniques. We conducted several tasks to optimize the classification system, such as.

- Exploring different keypoint detectors: SIFT, Dense SIFT and AKAZE (with different values of step size for the Dense SIFT).
- Applying of a norm or scaler for the features.
- Experimenting with different amounts of codebook sizes.
- Testing several classifiers and otimizing their hyperparameters: k-nn (number of neighbours and metric), SVM (Kernel, degree of the polynomial and C value) and Logistic Regression (penalty and C value).
- Incorporating dimensionality reduction - PCA.
- Implementing cross-validation to validate the different combinations and methods.
- Using spatial pyramids to extract the features of the images.
- Employing Fisher vectors to define the codebook.


All the hyperparameters are optimized using wandb.ai.
