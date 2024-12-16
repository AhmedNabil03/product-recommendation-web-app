# Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns

# Importing Data

data = pd.read_csv('walmart_product_review.tsv', sep='\t')

# Data Cleaning

df = data.copy()
df.head(1)

df.shape

df.isnull().sum()

columns_to_drop = df.columns[df.isnull().sum() >= df.shape[0] * 0.9]
columns_to_drop

df = df.drop(columns=columns_to_drop)
df.head(1)

df.isnull().sum()

df['Crawl Timestamp']

df['Product Company Type Source'].value_counts()

df['Retailer'].value_counts()

df['Market'].value_counts()

df['Product Currency'].value_counts()

df['Product Available Inventory'].value_counts()

df['Joining Key']

df.drop(columns=['Crawl Timestamp', 'Product Company Type Source', 'Retailer', 'Market', 'Product Currency', 'Product Available Inventory', 'Joining Key'], inplace=True)
df.head(2)

column_name_mapping = {
    'Uniq Id': 'ID',
    'Product Id': 'ProdID',
    'Product Category': 'Category',
    'Product Brand': 'Brand',
    'Product Name': 'Name',
    'Product Price': 'Price',
    'Product Url': 'URL',
    'Product Description': 'Description',
    'Product Image Url': 'AllImagesURLs',
    'Product Tags': 'Tags',
    'Product Rating': 'Rating',
    'Product Reviews Count': 'ReviewCount'
}
df.rename(columns=column_name_mapping, inplace=True)

df.head(2)

df.isnull().sum()

fill_values = {
    'Rating': 0,
    'ReviewCount': 0,
    'Category': '',
    'Brand': '',
    'Description': ''
}

df.fillna(fill_values, inplace=True)
df.isnull().sum()

df.duplicated().sum()

df.columns

# EDA

print('Num of Unique Products: ', df['ProdID'].nunique())
print('Num of Brands: ', df['Brand'].nunique())

print('Totlal num of categories: ', df['Category'].nunique())

highest_level_cat = df['Category'].str.strip().str.split(' > ').str[0].unique()
print('Highest Level Categories: ', len(highest_level_cat))
print('Highest Level Categories: ', df['Category'].str.split(' > ').str[0].unique())

df[df['Price'] != 0][['Price']].describe()

sns.histplot(df['Price'], kde=True)

df[df['Rating'] != 0][['Rating']].describe()

df['Rating'].plot(kind='hist', bins=20)

df[df['ReviewCount'] != 0][['ReviewCount']].describe()
# alot of zero ratings for items that haven't been rated

df['ReviewCount'].plot(kind='kde')

# Data Preprocessing

# the last 2 tags are always the same
df['Tags'].apply(lambda x: [tag.strip() for tag in x.split(',')][-2:]).value_counts()

# remove the last 2 tags
df['Tags'] = df['Tags'].apply(lambda x: ', '.join([tag.strip() for tag in x.split(',')][:-2]))

# text clean function
import re
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s,]', '', text)
    text = text.lower()
    return text

df['Description'] = df['Description'].apply(clean_text)
df['Tags'] = df['Tags'].apply(clean_text)
df['Category'] = df['Category'].apply(clean_text)

df['ImageURL'] = df['AllImagesURLs'].apply(lambda x: x.split()[0])

df.to_csv('cleaned_products.csv')

# Base Recommendation System

### A Base Recommender Based on Rating and Review Count

def base_recommender(df, num_items=10):
    top_rated = df.sort_values(by=['Rating', 'ReviewCount'], ascending=False)
    top_rated = top_rated.head(num_items)
    return top_rated[['Name', 'Rating', 'ReviewCount', 'Price', 'URL']]

recommended_items = base_recommender(df, num_items=10)
recommended_items

# Items with fewer reviews often have inflated ratings, while highly reviewed items have more reliable ratings but can be overlooked.
# Combining ratings and review counts balances quality (ratings) with popularity (review counts), ensuring fairer recommendations.
# A weighted recommender addresses inflated ratings for low-reviewed items and highlights popular, reliable products.

### A Weighted Recommender Based on Rating and Review Count

def weighted_recommender(df):
    df['NormRating'] = df['Rating'] / df['Rating'].max()
    df['NormReview'] = np.log1p(df['ReviewCount']) / np.log1p(df['ReviewCount']).max()
    df['Score'] = (0.9 * df['NormRating'] + 0.1 * df['NormReview'])

    top_items = df.sort_values(by='Score', ascending=False)
    return top_items.head(6)

recommended_items = weighted_recommender(df)
recommended_items[['Name', 'Rating', 'ReviewCount', 'Price', 'Score']]

# Content-Based Recommendation System

df['CombinedText'] = df['Category'] + ' ' + df['Tags']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['CombinedText'])

tfidf_matrix.shape

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim

def recommend_product(product_index, cosine_sim=cosine_sim, top_n=10):
    sim_scores = list(enumerate(cosine_sim[product_index]))    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    product_indices = [x[0] for x in sim_scores]
    similarity_scores = [x[1] for x in sim_scores]
    
    recommended_products = df.iloc[product_indices]
    recommended_products['Similarity'] = similarity_scores
 
    return recommended_products.head(6)

recommended_products = recommend_product(product_index=10)
recommended_products[['Name', 'Rating', 'ReviewCount', 'Price', 'Similarity']]




