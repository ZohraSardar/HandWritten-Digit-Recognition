# Handwritten Digit Recognition with CNN

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)  
**A TensorFlow/Keras-based CNN model to classify handwritten digits from the MNIST dataset.**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🌟 Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to recognize handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model achieves high accuracy in classifying 28x28 grayscale images of handwritten digits.

---

## 🚀 Features
- **CNN Architecture**: Includes `Conv2D`, `MaxPooling2D`, and `Dense` layers.
- **Dataset**: Uses the MNIST dataset (60,000 training + 10,000 test images).
- **Training Pipeline**: Adam optimizer and sparse categorical cross-entropy loss.
- **Evaluation**: Accuracy metrics on the test set.
- **Prediction Script**: Test the model with the MNIST test dataset.

---

## ⚙️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition







**Install dependencies**:

bash
pip install tensorflow numpy matplotlib
🖥️ Usage
1. Train the Model
Run the training script directly from the Jupyter notebook Hand Written Digit Recognition.ipynb:

python
model.fit(x_trainr, y_train, epochs=5, validation_split=0.3)
The trained model weights will be saved implicitly in the notebook environment.

2. Evaluate the Model
Evaluate performance on the test set:

python
test_loss, test_acc = model.evaluate(x_testr, y_test)
print(f"Test Accuracy: {test_acc}")
3. Make Predictions
Predict a digit from the test set:

python
predictions = model.predict(x_testr)
print(np.argmax(predictions[0]))  # Replace index to test other images
4. Visualization
Visualize test images and predictions using matplotlib:

python
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
📊 Results
Test Accuracy: ~98.3% on the MNIST test set.

Sample Prediction:
Prediction Example
Model prediction for digit "7" with high confidence.

📂 Project Structure
├── Hand Written Digit Recognition.ipynb  # Jupyter notebook with code
├── README.md                             # Project documentation
└── requirements.txt                      # Dependencies (TensorFlow, NumPy, Matplotlib)
