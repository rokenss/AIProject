# AIProject

Comparison of Tree-Based Methods and Neural Networks for Face Detection 

1. Introduction 

In this project, we explored and compared different machine learning methods for face detection. The goal was to see how tree-based models like Decision Trees and Random Forests stack up against Neural Networks when working with image-based data. We tested several feature sets and measured each model’s performance based on accuracy and how long it took to train them. For all experiments, we used the famous48 dataset. 

2. Feature Sets  

Several feature engineering approaches were investigated: 

- Raw Pixels: The models were trained directly on the raw pixel values of the images. 

- Handcrafted Features: A set of features was manually engineered. These included statistical measures like mean, standard deviation, skewness, and kurtosis, as well as image-specific features like Sobel gradients, symmetry, and grid means. 

- Combined Features: This set combined the top 30 most important features identified by a Random Forest model (trained on raw pixels) with the handcrafted features. 

- Extended Features: A more comprehensive feature set was created by concatenating raw pixels with several advanced image descriptors: gradient angles, Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), Haar-like features, and LAB colour space-inspired features. The final feature vector shape for this set was (6835, 2078). 

 

 

3. Models Evaluated 

The following models were trained and evaluated: 

- Decision Tree (DT): A basic tree-based model from Scikit-learn. 

- Random Forest (RF): An ensemble model using multiple decision trees.  

- One version was implemented using Scikit-learn in model.ipynb. 

- Another, more tuned version (e.g., 400 estimators, max depth 60) was also implemented using Scikit-learn in pytorch.py with extended features. 

- Neural Network (NN):  

- An MLPClassifier (Multi-layer Perceptron) from Scikit-learn was used in model.ipynb. 

- A custom SimpleNN was built using PyTorch in pytorch.py, featuring linear layers with ReLU activations and Dropout. 

 

4. Results 

	Scikit-learn models: 

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


 Models with Extended Features: 

 - Feature Set: Raw Pixels + Gradient Angles + LBP + HOG + Haar + LAB features. Feature shape: (6835, 2078). 
 - Random Forest:  
   	- Accuracy = 0.7485 
   	- Time = 160.09s
- Neural Network (PyTorch - SimpleNN):  
  	- The model was trained for 50 epochs. 
	- Maximum Test Accuracy achieved: 0.8928 (at Epoch 49). 

5. Discussion 

	To be continued... 

 
