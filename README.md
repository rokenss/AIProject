Face Recognition Pipeline Report (RF, FFNN, CNN) 

 

This project implements a comprehensive face recognition pipeline using the famous48 dataset. It extracts handcrafted features from 24×24 grayscale images and applies three different machine learning approaches: 
 - Random Forest (RF) 
 - Feedforward Neural Network (FFNN) 
 - Convolutional Neural Network (CNN on raw pixels) 

1. Dataset & Feature Extraction 

  - famous48 is a set of example images contained faces of 48 famous persons like sportsmen, politicians, actors or television stars. It was divided into 3 files: x24x24.txt, y24x24.txt, z24x24.txt, each containing 16 personal classes. In total 6835 images. 
  - Each image: 24×24 grayscale → 576 pixels 
  - Features extracted: 
        -- Raw Pixels 
        -- Gradient Angles (Sobel filter) 
        -- LBP (Local Binary Patterns) 
        -- HOG (Histogram of Oriented Gradients) 
        -- Haar-like rectangle features (via integral image) 
        -- LAB (Local Grouped Binary patterns) 
  - Final feature shape: (6835, 2078) 

2. Libraries and Dependencies 

  - numpy, scipy: numerical operations and filtering   
  - scikit-learn: preprocessing, RandomForestClassifier, model evaluation   
  - torch (PyTorch): model training and optimisation, FFNN, CNN   
  - skimage: feature extraction (HOG, LBP)   

3. Random Forest Classifier 

  - Trained on extracted features 
  - Parameters: n_estimators=400, max_depth=60, max_features='sqrt' 
  - Accuracy: 76.02% 
  - Training Time: 59.65s 

4. Feedforward Neural Network (PyTorch) 
 
  - Input: Extracted features (2078-dimensional) 
  - Architecture: 
       -- Linear → ReLU → Dropout(0.3) 
       -- Linear → ReLU → Dropout(0.3) 
       -- Output → num_classes 
  - Regularization: Dropout, weight_decay=1e-4 
  - Optimizer: Adam (lr=0.001) 
  - Epochs: 50 
  - Accuracy: 90.06% 
  - Training time: 4.21s 

5. Convolutional Neural Network (CNN) 

  - Input: Raw 24×24 grayscale images 
  - Architecture: 
      -- Conv2D → BN → ReLU → MaxPool 
      -- Conv2D → BN → ReLU → MaxPool 
      -- Flatten → Linear → Dropout → Output 
  - Regularization: Dropout, BatchNorm, weight_decay=1e-4 
  - Optimizer: Adam (lr=0.001) 
  - Epochs: 30 
  - Accuracy: 87.91% 
  - Training time: 13.76s 

6. Observations
   
   - FFNN outperformed CNN and RF on this task due to the rich, handcrafted features. 
   - CNN required more epochs to converge but reached near parity with FFNN. 
   - Random Forest offered strong baseline performance with less tuning. 
   - Feature engineering played a critical role in boosting accuracy beyond 90%. 

7. Conclusion 

 Combining classic feature engineering with modern neural networks provides a powerful approach to face recognition, especially when data is limited. CNNs perform well directly on pixels, but FFNNs trained on strong descriptors can outperform even deep models on small images like 24×24. 
