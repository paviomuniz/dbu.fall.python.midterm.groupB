
Project Proposal: Analyzing News Sentiment vs. Google Stock Price Movements
1. Project Title
Predicting Stock Price Movements Using News Sentiment Analysis (Google Stock Dataset)

2. Project Overview
The goal of this project is to analyze whether news sentiment has a measurable impact on Google’s stock price. Using historical stock data and financial news articles, we will calculate sentiment scores and study how they relate to real stock price movement.
We will also attempt to build a simple machine learning model to predict short-term price direction (up or down) based on sentiment.

3. Datasets to Be Used
1.	Google Daily Stock Prices (2004–Today)
https://www.kaggle.com/datasets/emrekaany/google-daily-stock-prices-2004-today
2.	Google Financial News Dataset (optional / based on availability)
https://www.kaggle.com/datasets/emrekaany/google-googl-financial-news-from-2000-to-today/

4. Scope of Work
This is what we intend to accomplish:
A. Data Collection & Cleaning
•	Load stock price dataset (Open, Close, High, Low, Volume).
•	Load financial news dataset (news title, date).
•	Clean text data (remove punctuation, stopwords, special characters).
•	Align news dates with corresponding stock trading days.

B. Sentiment Analysis
•	Use a Python NLP library (VADER, TextBlob, or HuggingFace model).
•	Generate sentiment scores for each news headline/article.
•	Categorize sentiment as positive, neutral, or negative.

C. Exploratory Data Analysis
We will analyze:
•	Does positive news correlate with price increases?
•	Does negative news correlate with price drops?
•	How strong is the correlation between sentiment score and daily returns?
•	Plot sentiment vs. stock price over time.

D. Predictive Modeling (Simple ML)
We will build a basic model to predict price movement (Up/Down) using:
•	Sentiment score
•	Lagged stock returns
•	Volume
•	Other simple features
Possible models:
•	Logistic Regression
•	Random Forest Classifier
•	Support Vector Machine
Goal: Predict whether the next day’s price will go up or down based on sentiment.

E. Deliverables
Our final project deliverables will include:
•	A Python script/Visual Studio explaining each step.
•	Visualizations (sentiment over time, correlation graphs, prediction accuracy).
•	A short report summarizing the findings, challenges, and predictive results.

5. Expected Outcome
We expect to determine:
•	Whether strong positive sentiment leads to upward price movement.
•	Whether negative sentiment predicts a drop.
•	How accurately a simple model can predict price direction based on sentiment.
The project does not aim to build a real trading system—just to explore the relationship between news sentiment and stock performance using Python.

6. Tools & Technologies
•	Python
•	Pandas, NumPy
•	NLTK / VADER / TextBlob (for sentiment)
•	Matplotlib / Seaborn
•	Scikit-learn

