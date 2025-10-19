Pneumonia Detection from Chest X-Rays
Project Description
This project compares traditional machine learning and deep learning approaches for detecting pneumonia from chest X-ray images. The study implements 7 different experiments using Scikit-learn and TensorFlow to find the most effective method for automated medical diagnosis.

Dataset
Source: Chest X-Ray Images (Pneumonia) from Kaggle

Images: 5,863 chest X-rays

Classes: Normal (1,583) and Pneumonia (4,273)

Note: Dataset consists of pediatric patients from Guangzhou Women and Children's Medical Center

Notebook Setup Instructions
1. Prerequisites
Google Colab (recommended) or Jupyter Notebook

Python 3.8+

GPU access (recommended for deep learning models)

2. Installation
python
# Run these commands in Colab or terminal
!pip install tensorflow==2.12.0
!pip install scikit-learn==1.2.2
!pip install opencv-python matplotlib seaborn pandas numpy
3. Data Setup
Download dataset from Kaggle: "Chest X-Ray Images (Pneumonia)"

Upload to your Google Drive or local environment

Update the path in the notebook's data loading section

4. Running the Notebook
The notebook Pneumonia_Detection_Full_Pipeline.ipynb is organized sequentially:

Data Loading & Exploration

Loads and examines the dataset

Shows sample images from both classes

Data Preprocessing

Resizes images (224x224 for DL, 128x128 for traditional ML)

Normalizes pixel values

Applies data augmentation (rotation, flipping, zoom)

Traditional Machine Learning

HOG feature extraction

Models: Logistic Regression, Random Forest, SVM

Deep Learning

Sequential CNN (3 convolutional layers)

Functional CNN (parallel convolutional branches)

Optimized with tf.data pipeline

Evaluation & Visualization

Performance metrics table

Learning curves

Confusion matrices

ROC curves

Grad-CAM explainability maps

Key Results
Model	Accuracy	Precision	Recall	F1-Score	AUC
SVM + HOG	87.2%	86.0%	88.5%	87.2%	0.913
Functional CNN	94.7%	94.0%	95.5%	94.7%	0.973
Experiments Conducted
Logistic Regression + HOG features

Random Forest + HOG features

SVM + HOG features

Random Forest + VGG16 deep features

CNN (Sequential API)

CNN (Functional API) - BEST PERFORMING

Optimized CNN + tf.data pipeline

Files Included
Pneumonia_Detection_Full_Pipeline.ipynb - Complete working notebook

requirements.txt - Python dependencies

Final_Report.pdf - Detailed project report

Presentation video (link in report)

Reproducibility
All random seeds are set (42)

Complete experiment logging

Step-by-step comments in notebook

Saved model weights available

Author
Daniel Kudum - African Leadership University

Note
This project was completed for the Introduction to Machine Learning course. The dataset is publicly available on Kaggle and all code is provided for reproducibility.
