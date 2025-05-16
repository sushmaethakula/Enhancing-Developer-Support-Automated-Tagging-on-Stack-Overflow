import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Streamlit Page Settings
st.set_page_config(layout="wide")
st.title("📊 Tag Prediction Data Exploration")
st.markdown("Visualizing the most common tags, text lengths, and more from the dataset.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('Final_Data.csv')
    df['tags'] = df['tags'].apply(eval)  # Convert stringified list to list

    # Ensure 'Text' column exists
    if 'Text' not in df.columns:
        df['Text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

    return df

df = load_data()

# --- Top 20 Most Frequent Tags ---
st.subheader("🔝 Top 20 Most Common Tags")
tag_counts = pd.Series([tag for tags in df['tags'] for tag in tags]).value_counts()

fig1, ax1 = plt.subplots(figsize=(12, 8))
tag_counts[:20].plot(kind='barh', ax=ax1)
ax1.set_title('Top 20 Most Frequent Tags')
ax1.set_xlabel('Count')
ax1.set_ylabel('Tag')
st.pyplot(fig1)

# --- Tags per Question ---
st.subheader("🏷️ Distribution of Number of Tags per Question")
df['num_tags'] = df['tags'].apply(len)
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.countplot(x='num_tags', data=df, ax=ax2)
ax2.set_title('Number of Tags per Question')
st.pyplot(fig2)

# --- Text Length ---
st.subheader("📝 Text Length Distribution")
df['text_len'] = df['Text'].apply(lambda x: len(str(x).split()))
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.histplot(df['text_len'], bins=50, kde=True, ax=ax3)
ax3.set_title('Text Length (Word Count)')
ax3.set_xlabel('Number of Words')
st.pyplot(fig3)

# --- Word Cloud ---
st.subheader("☁️ Word Cloud of All Questions")
all_text = ' '.join(str(text) for text in df['Text'])
clean_text = re.sub(r'[^a-zA-Z\s]', '', all_text)  # Remove symbols from word cloud too
wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(clean_text)

fig5, ax5 = plt.subplots(figsize=(15, 7))
ax5.imshow(wordcloud, interpolation='bilinear')
ax5.axis('off')
st.pyplot(fig5)
