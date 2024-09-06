Predicting DoorDash ETA Using Machine Learning

## Authors
Janak Sunil, Alan Meyer, Rabbun Ishmam Haider, Zhirong Lu, Christopher Payan

## Abstract
This project aims to develop machine learning models to accurately predict food delivery duration for DoorDash based on factors such as store information, order details, and delivery partner availability. We preprocess a real-world dataset, engineer relevant features, and compare the performance of polynomial regression and neural network models.

## Introduction
Online food delivery platforms have seen explosive growth in recent years, accelerated by the COVID-19 pandemic. Accurately predicting delivery duration is essential for setting realistic customer expectations, optimizing delivery fleet utilization, and identifying bottlenecks in the delivery process.

## Problem Formulation
Given a dataset of historical food delivery orders, our goal is to train a machine learning model that accurately predicts the delivery duration for new, unseen orders. We formulate this as a regression problem, aiming to minimize the expected squared loss.

## Data Description
The dataset contains 193,806 historical food delivery orders from DoorDash, with 12 features including store information, order details, and delivery partner availability. The target variable is the actual delivery duration in minutes.

## Methodology
1. **Data Preprocessing**: 
   - Timestamp conversion and duration calculation
   - Missing value handling
   - Feature-target separation
   - Categorical encoding and numeric scaling
   - Outlier removal

2. **Polynomial Regression**:
   - Trained models of degrees 1 to 4
   - Evaluated using RMSE and R-squared

3. **Neural Network**:
   - Deep neural network using PyTorch
   - Architecture: Embedding layer, 9 fully connected hidden layers, SiLU activation, batch normalization, residual connections, dropout, and L2 regularization

## Results and Discussion
- Polynomial Regression: Degree 2 model performed best (Test RMSE: 13.06, R-squared: 0.178)
- Neural Network: Severe overfitting (Train R-squared: 0.9984, Test R-squared: -1.0470)
- Both models showed limitations in capturing all relevant factors affecting delivery duration

## Conclusion
While our initial models provide valuable insights, there is substantial room for improvement in predicting food delivery duration. Future work should focus on:
1. Data Enhancement: Collecting more diverse data, incorporating real-time information
2. Feature Engineering: Creating informative features, extracting temporal information
3. Model Refinement: Tuning hyperparameters, implementing early stopping, exploring alternative architectures
4. Operational Integration: Developing a system for continuous model updating and integration into the delivery platform


## References
1. Suryhaa, Dharun. "DoorDash ETA Prediction" Kaggle. [Dataset Link](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction/data)
2. Project code on Google Colab: [Colab Notebook](https://colab.research.google.com/drive/1XonEjYetLa6NodhsK6lPAQdecyOPsy9K?usp=sharing)
