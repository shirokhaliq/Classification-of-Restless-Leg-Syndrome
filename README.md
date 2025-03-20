# Anxiety Detection Through Restless Leg Syndrome

## Overview
This project focuses on classifying Restless Leg Syndrome (RLS) to assist in anxiety detection using accelerometer data and machine learning techniques. Data was collected through leg movement recordings under different activities, processed, and analyzed using various models to identify RLS patterns effectively.

## Table of Contents
- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [References](#references)

## Project Objectives
1. Collect and preprocess accelerometer data capturing leg movements from RLS and non-RLS individuals.
2. Extract relevant statistical, motion, and frequency-based features.
3. Develop and evaluate machine learning models to classify RLS and non-RLS patterns.
4. Contribute towards anxiety detection by leveraging RLS patterns.

## Dataset
- **Participants:** 3 RLS and 3 non-RLS participants
- **Activities Recorded:**
  - Sitting Still
  - Watching Horror Movies
  - Playing Piano
- **Data Source:** Smartphone app (Advanced Physical Toolbox)
- **Focus:** Z-axis accelerometer data (leg movements)

## Methodology
- **Preprocessing:** Low-pass filtering (Butterworth filter), outlier removal
- **Feature Extraction:**
  - Statistical Features (Mean, Median, Skewness, etc.)
  - Motion Features (Zero Crossings, Angular Velocity, etc.)
  - Fourier-transformed Features (Dominant Frequency, PSD Entropy, etc.)
- **Feature Selection:** PCA & Random Forest Importance
- **Clustering:** K-Means for label assignment
- **Balancing:** Random Under Sampling (RUS)
- **Modeling:**
  - Naïve Bayes
  - K-Nearest Neighbors
  - Support Vector Machine
  - Multilayer Perceptron (MLP)

## Features
Extracted 30 features categorized into:
- **Statistical Features (10)**
- **Motion Features (11)**
- **Frequency Features (9)**

## Machine Learning Models
Implemented & evaluated:
- Naïve Bayes (GaussianNB)
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Results
- **MLP outperformed other models** with ~98%+ accuracy across all activities.
- **SVM performed well**, though slightly lower.
- **NB & KNN showed signs of underfitting** in some activities.


## Future Work
- Integration of larger and more diverse datasets
- Exploration of deep learning models (e.g., CNN, LSTM)
- Real-time deployment on wearable devices for anxiety monitoring
- Incorporate multimodal physiological data (heart rate, ECG)

## References
- Full reference list can be found in the `FinalReport.pdf` or `references/` folder.

