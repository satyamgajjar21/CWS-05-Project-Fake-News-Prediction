## Fake News Prediction Using Machine Learning

## Project Overview
This project implements a machine learning based system to identify fake news articles using natural language processing techniques. The model learns patterns from labeled news text data and predicts whether a given news article is real or fake.

## Problem Statement
Fake news spreads misinformation rapidly and affects public opinion and decision making. Manual verification is slow and unreliable at scale. This project aims to automate fake news detection using supervised machine learning.

## Dataset Description
The dataset contains news articles with two main components:
Text data representing the news content
Labels indicating whether the news is fake or real

## Technologies Used
Python  
Jupyter Notebook  
Pandas  
NumPy  
Scikit learn  
NLTK  
TF IDF Vectorizer  

## Project Workflow
Data loading and exploration  
Data cleaning and preprocessing  
Text vectorization using TF IDF  
Model training  
Model evaluation  
Prediction on new data  

## Text Preprocessing
Lowercasing text  
Removing punctuation and special characters  
Removing stopwords  
Tokenization  

## Feature Extraction
TF IDF Vectorizer is used to convert textual data into numerical feature vectors suitable for machine learning models.

## Machine Learning Models
Logistic Regression  
Passive Aggressive Classifier  

## Model Training
The dataset is split into training and testing sets. Models are trained using processed text features to learn patterns associated with fake and real news.

## Model Evaluation
Model performance is evaluated using accuracy score, confusion matrix, and classification report to analyze prediction quality.

## Results
The trained model achieves strong accuracy in detecting fake news and generalizes well on unseen data.

## How To Run The Project
Clone the repository  
Install required Python libraries  
Open the Jupyter Notebook  
Run all cells sequentially  

## Project Structure
Fake News Prediction.ipynb  
Dataset files  
README.md  

## Future Improvements
Use deep learning models such as LSTM or Transformers  
Expand dataset size  
Improve preprocessing with lemmatization  
Deploy as a web application  

## Conclusion
This project demonstrates practical use of machine learning and NLP to address the real world problem of fake news detection.

## Author
Satyam Gajjar
