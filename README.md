# Project

## Task1 work
The main goal of this week is to develop an image classification system with handcrafted techniques. We conducted several tasks to optimize the classification system, such as.

- Exploring different keypoint detectors: SIFT, Dense SIFT and AKAZE (with different values of step size for the Dense SIFT).
- Applying of a norm or scaler for the features.
- Experimenting with different amounts of codebook sizes.
- Testing several classifiers and otimizing their hyperparameters: k-nn (number of neighbours and metric), SVM (Kernel, degree of the polynomial and C value) and Logistic Regression (penalty and C value).
- Incorporating dimensionality reduction - PCA.
- Implementing cross-validation to validate the different combinations and methods.
- Using spatial pyramids to extract the features of the images.
- Employing Fisher vectors to define the codebook.


All the hyperparameters are optimized using wandb.ai.

The results of the evalation metrics for all the combinations of classifier-descriptor are summarized in the following table:


|CLASSIFIER + DESCRIPTOR|AUC|ACCURACY|AVERAGE F1|AVERAGE PRECISON|AVERAGE RECALL|
|:----|:----|:----|:----|:----|:----|
|Logistic  + DenseSIFT|0.98|0.82|0.82|0.82|0.82|
|SVM  + DenseSIFT|0.97|0.81|0.81|0.81|0.81|
|KNN + DenseSIFT|0.96|0.79|0.79|0.79|0.79|
|Logistic  + SIFT|0.95|0.72|0.72|0.72|0.72|
|Logistic  + AKAZE|0.95|0.69|0.70|0.70|0.69|
|SVM  + SIFT|0.94|0.67|0.68|0.68|0.68|
|SVM  + AKAZE|0.93|0.64|0.64|0.65|0.64|
|KNN  + SIFT|0.92|0.64|0.65|0.66|0.64|
|KNN  + AKAZE|0.88|0.59|0.61|0.62|0.60|


The best combination of optimized values of the hyperparameters for each combination are shown in the following table:

|CLASSIFIER + DESCRIPTOR|C|DEGREEE POLYNOMIAL|FISHER VECTOR|KERNEL|LEVEL PYRAMID|N COMPONENTS|N FEATURES|N NEIGHBORS|N WORDS|P|SCALING|STEP SIZE|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|Logistic  + DenseSIFT|0.1|-|false|-|3|59|1024|-|223|-|true|4|
|SVM  + DenseSIFT|0.01|5|false|Histogram Intersection|4|54|1024|-|314|-|false|12|
|KNN + DenseSIFT|-|-|false|-|1|51|1024|9|890|2|false|18|
|Logistic  + SIFT|0.001|-|true|-|1|53|1024|-|132|-|true|-|
|Logistic  + AKAZE|0.01|-|true|-|3|62|1024|-|203|-|true|-|
|SVM  + SIFT|1.00|2|false|RBF|1|56|1024|-|239|-|true|-|
|SVM  + AKAZE|0.01|4|false|Histogram Intersection|2|24|1024|-|478|-|true|-|
|KNN  + SIFT|-|-|false|-|3|39|1024|25|297|2|true|92|
|KNN  + AKAZE|-|-|false|-|1|52|1024|8|869|2|true|82|



## Task2 work
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

This week's results can be summarized in the following table:




## Task3 work

## Task4 work



(State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.)