# Multiclass-Fish-Image-Classification


# 🐟 Fish Classification App

A Deep Learning-based application to classify different species of fish using Convolutional Neural Networks (CNNs) and transfer learning models.  
Built with **TensorFlow**, **Keras**, and **Streamlit**, this app allows users to upload an image of a fish and get an instant prediction along with model confidence.

---

## 🚀 Features

- Upload fish images (`.jpg`, `.jpeg`, `.png`)
- Predicts the fish species using a trained deep learning model
- Displays:
  - 🧠 Top-3 most likely predictions
  - 📊 Confidence distribution for all classes
  - 💾 Downloadable prediction report (CSV)
- Deployed with **Streamlit** for an interactive UI

---

## 🧩 Approach

### 1️⃣ Data Preprocessing and Augmentation
- Rescaled images to the `[0, 1]` range
- Applied augmentation techniques:
  - Rotation
  - Zoom
  - Horizontal & vertical flipping  
- Split dataset into training, validation, and test sets

### 2️⃣ Model Training
- Trained a custom **CNN model** from scratch
- Experimented with **five pre-trained models**:
  - VGG16  
  - ResNet50  
  - MobileNet  
  - InceptionV3  
  - EfficientNetB0  
- Fine-tuned each pre-trained model on the fish dataset
- Saved the **best-performing model** (based on accuracy) in `.h5` or `.pkl` format

### 3️⃣ Model Evaluation
- Compared metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
- Generated confusion matrices for each model
- Visualized training & validation accuracy/loss

### 4️⃣ Deployment
- Deployed the model using **Streamlit**
- Features:
  - Image upload & preview
  - Real-time classification
  - Model confidence display
  - Confidence distribution chart
  - CSV report download option

---

## 🧠 Example Output

<img width="352" height="845" alt="image" src="https://github.com/user-attachments/assets/59539b46-3096-4058-a359-ee31fa9e9cce" />


**Sample Prediction:**
- **Final Prediction:** `fish_sea_food_black_sea_sprat`
- **Confidence:** `13.67%`


