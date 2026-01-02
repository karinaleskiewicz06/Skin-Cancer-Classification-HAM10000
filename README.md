# Skin Cancer Classification using Deep Learning (PyTorch)

## 🔬 Project Overview
This project was developed as part of the **HerStory** mentoring program. The goal was to build a Convolutional Neural Network (CNN) capable of classifying skin lesions into 7 different diagnostic categories using the **HAM10000** dataset.

The main focus of this project was to handle a multi-class medical classification task while actively combating **overfitting** through regularization techniques.

## 📊 Dataset
The model utilizes the **HAM10000** ("Human Against Machine with 10000 training images") dataset.
- **Total Images:** 10,015 dermoscopic images.
- **Classes:** 7 (Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions, Dermatofibroma, Melanoma, Melanocytic nevi, Vascular lesions).
- **Preprocessing:** Images were resized to 128x128 pixels and normalized for efficient training on GPU.

## 🏗️ Model Architecture & Hyperparameters
I designed a custom CNN architecture with the following specifications:
- **Layers:** 3 Convolutional layers followed by Max Pooling and ReLU activation.
- **Regularization:** Integrated a **Dropout layer (0.5)** before the fully connected layers to ensure the model generalizes well to unseen data.
- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Adam with a learning rate of `1e-4`.
- **Batch Size:** 16.

## 📈 Results
The model achieved a stable performance after 25 epochs:
- **Test Accuracy:** **74.35%**
- **Training vs. Validation:** The learning curves (Loss and Accuracy) show strong convergence with no significant gap, proving that the **Dropout** implementation successfully prevented overfitting.
- 
<img width="3758" height="1552" alt="acc and loss" src="https://github.com/user-attachments/assets/8bbb3efc-264e-4974-908e-38b7f31cb457" />



## 🚀 How to Run
1. Clone the repository.
2. The dataset is automatically downloaded via the `opendatasets` library (Kaggle credentials required).
3. Install dependencies: `pip install torch torchvision pandas matplotlib scikit-learn Pillow`.
4. Run the `skin_cancer.ipynb` notebook.

## 📝 Key Takeaways
- Successfully transitioned from simple binary classification to a complex 7-class medical diagnostic task.
- Managed large-scale medical data on GPU.
- Demonstrated the importance of regularization (Dropout) in medical AI to ensure reliable diagnostics.
