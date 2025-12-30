# Rice_leaf_disese_detection-using-CNN

## Problem Statement
Rice is a staple crop, and early detection of leaf diseases is critical for improving yield and reducing crop loss.  
This project aims to classify rice leaf images into healthy and diseased categories using a Convolutional Neural Network (CNN).

---

## Dataset Overview
- Image dataset containing rice leaf samples
- Organized into class-wise folders
- Images are used to train a deep learning model for disease classification

---

##  Approach
1. **Data Preparation**
   - Extracted and organized image data from zip files
   - Resized and normalized images for model training

2. **Data Augmentation**
   - Applied image transformations to improve generalization
   - Reduced overfitting using augmentation techniques

3. **Model Architecture**
   - Built a CNN using TensorFlow and Keras
   - Included convolution, pooling, and dense layers

4. **Model Training**
   - Trained the CNN on labeled rice leaf images
   - Used validation data to monitor performance

5. **Model Evaluation**
   - Evaluated model performance on unseen data
   - Visualized training and validation accuracy/loss

---

##  Model Used
- Convolutional Neural Network (CNN)
- Built using TensorFlow (Keras API)

---

##  Results
- Successfully trained a CNN model for rice leaf disease classification
- Model demonstrated good learning behavior with improving accuracy over epochs
- Training and validation metrics were visualized to assess performance

---

##  Key Insights
- CNNs are effective for image-based plant disease detection
- Image preprocessing and augmentation significantly improve model performance
- Proper dataset organization is crucial for training deep learning models

---

##  Tech Stack
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow (Keras)  
- **Libraries:** NumPy, Matplotlib, Pillow, Scikit-learn  
- **Domain:** Computer Vision, Image Classification  

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Ranganath141/Rice-leaf-disease-detection-cnn.git

2. Navigate to the project directory:

    cd rice-leaf-disease-detection-cnn

3. Install dependencies:

    pip install -r requirements.txt

4.  Open and run the Notebook:

    Rice__leaf_disease_model.ipynb

## Future Improvements

Hyperparameter tuning for CNN architecture

Transfer learning using pre-trained models (VGG16, ResNet)

Deployment as a web or mobile application

Expand dataset for better generalization


👤 Author

Ranganath
Aspiring Data Scientist / AI-ML Engineer

GitHub: https://github.com/Ranganath141
