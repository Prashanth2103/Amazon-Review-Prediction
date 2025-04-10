# 18-11-2024
#### I'm still working on dashboard and will be updating the changes done on the day of presentation. There will be only minor changes.
# I'm using streamlit for creating dashboard


# importing libraries
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Loading dataset
def load_data():
    data = pd.read_csv(r"F:\Reviews.csv")
    data_sampled = data.sample(n=100000, random_state=42)
    data_sampled = data_sampled.dropna(subset=['Summary', 'Text'])
    data_sampled['Text'] = data_sampled['Text'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    data_sampled['Date'] = pd.to_datetime(data_sampled['Time'], unit='s')
    return data_sampled

data = load_data()

# Sidebar options
st.sidebar.title("Amazon Reviews Dashboard")
options = st.sidebar.selectbox("Choose a Visualization", 
                               ["Distribution of Scores", 
                                "Sentiment Distribution", 
                                "Helpfulness Ratio Distribution", 
                                "Monthly Review Score Trend", 
                                "Word Cloud",  
                                "Customer Segmentation"]) 
                                

# 1. Distribution of Scores
if options == "Distribution of Scores":
    st.title("Distribution of Review Scores")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Score', data=data, palette='viridis')
    plt.title("Distribution of Review Scores")
    st.pyplot(plt)

# 2. Sentiment Distribution
if options == "Sentiment Distribution":
    st.title("Sentiment Distribution")
    data['Sentiment'] = data['Score'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sentiment', data=data, palette='coolwarm')
    plt.title("Sentiment Distribution")
    st.pyplot(plt)

# 3. Helpfulness Ratio Distribution
if options == "Helpfulness Ratio Distribution":
    st.title("Helpfulness Ratio Distribution")
    data['HelpfulnessRatio'] = data.apply(lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] 
                                          if row['HelpfulnessDenominator'] > 0 else 0, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['HelpfulnessRatio'], bins=30, kde=True)
    plt.title("Helpfulness Ratio Distribution")
    st.pyplot(plt)

# 4. Monthly Review Score Trend
if options == "Monthly Score Trend":
    st.title("Monthly Average Score Trend")
    monthly_avg_score = data.resample('M', on='Date')['Score'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg_score.index, monthly_avg_score.values, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Average Score")
    plt.title("Monthly Average Score Trend")
    st.pyplot(plt)

# 5. Word Cloud
if options == "Word Cloud":
    st.title("Word Cloud of Reviews")
    text = " ".join(review for review in data['Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# 6. Customer Segmentation
if options == "Customer Segmentation":
    st.title("Customer Segmentation")
    customer_features = data.groupby('UserId').agg({'Score': 'mean', 'HelpfulnessNumerator': 'mean', 'ProductId': 'count'}).rename(columns={'ProductId': 'ReviewCount'})
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_features['Cluster'] = kmeans.fit_predict(customer_features)
    pca = PCA(n_components=2)
    components = pca.fit_transform(customer_features[['Score', 'HelpfulnessNumerator', 'ReviewCount']])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=customer_features['Cluster'], palette='viridis')
    plt.title("Customer Segmentation Clusters")
    st.pyplot(plt)



