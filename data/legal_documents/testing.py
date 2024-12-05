#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

def load_dataset(csv_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return df


# In[3]:


csv_path = "legal_text_classification.csv"
df = load_dataset(csv_path)


# In[4]:


def perform_eda(df):
    """
    Perform basic EDA on the dataset.
    """
    print("First 5 records:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nClass Distribution:")
    print(df['case_outcome'].value_counts())


# In[5]:


# perform_eda(df)


# In[6]:


df = df.dropna(subset=['case_text'])
# perform_eda(df)


# In[ ]:


df['case_id'] = df['case_id'].str.replace('Case', '').astype(int)
df['case_outcome'] = df['case_outcome'].astype(str)
df['case_title'] = df['case_title'].astype(str)
df['case_text'] = df['case_text'].astype(str)
# perform_eda(df)


# In[12]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """
    Clean the input text by:
    - Lowercasing
    - Removing special characters and digits
    - Removing stopwords
    - Lemmatizing
    """
    # Lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join back to string
    cleaned_text = ' '.join(tokens)
    return cleaned_text


# In[13]:


def preprocess_texts(df):
    """
    Apply text cleaning to 'case_title' and 'case_text'.
    """
    df['cleaned_title'] = df['case_title'].apply(lambda x: clean_text(str(x)))
    df['cleaned_text'] = df['case_text'].apply(lambda x: clean_text(str(x)))
    return df


# In[14]:


df = preprocess_texts(df)


# In[21]:


df['cleaned_text'].apply(lambda x: len(x.split())).describe()


# In[22]:


def split_into_chunks(text, max_length=512):
    """
    Split text into chunks of maximum 'max_length' words.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

def split_texts(df, max_length=512):
    """
    Apply text splitting to 'cleaned_text'.
    Each case can have multiple chunks.
    """
    df['text_chunks'] = df['cleaned_text'].apply(lambda x: split_into_chunks(x, max_length))
    return df


# In[23]:


df = split_texts(df)


# In[25]:


from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

def generate_embeddings(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Generate embeddings for each text chunk.
    """
    model = SentenceTransformer(model_name)
    all_chunks = df['text_chunks'].explode().tolist()
    
    print("Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    return all_chunks, embeddings


# In[ ]:


all_chunks, embeddings = generate_embeddings(df)

import pickle
all_chunks, embeddings = generate_embeddings(df)
with open('all_chunks.pkl', 'wb') as f:
    pickle.dump(all_chunks, f)
np.save('embeddings.npy', embeddings)
