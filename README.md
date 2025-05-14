# AIProject

Comparison of Tree-Based Methods and Neural Networks for Face Detection 

## 1. Introduction 

In this project, we explored and compared different machine learning methods for face detection. The goal was to see how tree-based models like Decision Trees and Random Forests stack up against Neural Networks when working with image-based data. We tested several feature sets and measured each model’s performance based on accuracy and how long it took to train them. For all experiments, we used the famous48 dataset. 

## 2. Feature Sets  

Several feature engineering approaches were investigated: 

- Raw Pixels: The models were trained directly on the raw pixel values of the images. 

- Handcrafted Features: A set of features was manually engineered. These included statistical measures like mean, standard deviation, skewness, and kurtosis, as well as image-specific features like Sobel gradients, symmetry, and grid means. 

- Combined Features: This set combined the top 30 most important features identified by a Random Forest model (trained on raw pixels) with the handcrafted features. 

- Extended Features: A more comprehensive feature set was created by concatenating raw pixels with several advanced image descriptors: gradient angles, Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), Haar-like features, and LAB colour space-inspired features. The final feature vector shape for this set was (6835, 2078). 

 

 

## 3. Models Evaluated 

The following models were trained and evaluated: 

- Decision Tree (DT): A basic tree-based model from Scikit-learn. 

- Random Forest (RF): An ensemble model using multiple decision trees.  

- One version was implemented using Scikit-learn in model.ipynb. 

- Another, more tuned version (e.g., 400 estimators, max depth 60) was also implemented using Scikit-learn in pytorch.py with extended features. 

- Neural Network (NN):  

- An MLPClassifier (Multi-layer Perceptron) from Scikit-learn was used in model.ipynb. 

- A custom SimpleNN was built using PyTorch in pytorch.py, featuring linear layers with ReLU activations and Dropout. 

 

## 4. Results 
### First version
### Scikit-learn models: 

| Feature Set           | Model           | Accuracy | Time   |
|-----------------------|------------------|----------|--------|
| Raw pixels            | Decision Tree    | 0.2376   | 9.15s  |
|                       | Random Forest    | 0.6554   | 26.07s |
|                       | Neural Network   | 0.8157   | 8.21s  |
| Handcrafted features  | Decision Tree    | 0.1978   | 0.46s  |
|                       | Random Forest    | 0.4119   | 4.70s  |
|                       | Neural Network   | 0.4664   | 7.39s  |
| Combined Features     | Decision Tree    | 0.2393   | 0.79s  |
|                       | Random Forest    | 0.5255   | 6.75s  |
|                       | Neural Network   | 0.6232   | 6.30s  |


 ### Models with Extended Features:
 ### Current version
 
A) Libraries and Dependencies  

 - numpy, scipy: numerical operations and filtering  
 - scikit-learn: preprocessing, model selection, RandomForestClassifier  
 - torch (PyTorch): model training and optimisation  
 - skimage: feature extraction (HOG, LBP)  

B) Data Loading  

Each .txt file contains flattened 24x24 grayscale face images and class labels. The loader function extracts pixels and the a3 class label from each sample.  

C) Feature Extraction  

Each image is represented using the following stacked features:  
 - Raw Pixels: 576 values  
 - Gradient Angles (Sobel Filters): 576 values  
 - LBP (Local Binary Patterns): 10-bin histogram  
 - HOG (Histogram of Oriented Gradients): shape features  
 - Haar-like Rectangle Features: region contrast via integral images  
 - LAB (Local Average Binary): binary patterns from 4x4 patches  

D) Standardisation  

All features are scaled using StandardScaler (mean=0, std=1), which helps improve neural network convergence and balance feature influence.  

E) Data Split  

The data is split into 85% train and 15% test using train_test_split with stratification.  

F) Random Forest Classifier  

Model: sklearn.ensemble.RandomForestClassifier  
 - n_estimators: 400 (number of trees)  
 - max_depth: 60 (limits overfitting)  
 - max_features: sqrt (recommended for classification)  
 - Trained on full feature set  

G) Results  

Example performance:  
 - Feature Shape: (6835, 2078)  
 - Random Forest Accuracy: ~76% Time: 56.79s 
 - Neural Network Accuracy: up to 89.8% Time: 4.21s 

H) Takeaways  
 - Raw + classic features (HOG, LBP, LAB) are effective without deep CNNs 
 

## 5. Discussion 

To be continued... 

 
