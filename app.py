import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from walmart_recommender import weighted_recommender, recommend_product

# Load your data (make sure to preprocess your dataset accordingly)
df = pd.read_csv('cleaned_products.csv')

def format_name(name, max_length=30):
    if len(name) > max_length:
        return name[:max_length] + "..."
    else:
        return name

products_names = sorted(df['Name'].unique(), reverse=False)

# Streamlit UI
st.title("Product Recommendation System")

product_name = st.selectbox(
    'Search for a product',
    [''] + products_names
)

if product_name:
    matching_products = df[df['Name'].str.contains(product_name, case=False)]

    if not matching_products.empty:
        # Find the product chosen by the user
        chosen_product = df[df['Name'].str.contains(product_name, case=False)].iloc[0]

        # Display card for the chosen product
        if product_name and not matching_products.empty:
            st.subheader("Selected Product")
            chosen_product_dict = chosen_product.to_dict()

            st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; align-items: center; background-color: #f9f9f9;">
                    <img src="{chosen_product_dict['ImageURL']}" width="100%" style="border-radius: 8px; object-fit: contain; max-height: 200px;" />
                    <div style="margin-top: 10px; text-align: center;">
                        <h3 style="color: #333;">{chosen_product_dict['Name']}</h3>
                        <p><strong>Price:</strong> ${chosen_product_dict['Price']}</p>
                        <p><strong>Rating:</strong> {chosen_product_dict['Rating']}<br><strong>Reviews:</strong> {chosen_product_dict['ReviewCount']}</p>
                        <p><a href="{chosen_product_dict['URL']}" target="_blank" style="text-decoration: none; color: white; background-color: #4CAF50; padding: 8px; border-radius: 4px;">Buy Now</a></p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Content-based recommendation (TF-IDF and Cosine Similarity)
        chosen_product_idx = df[df['Name'] == chosen_product['Name']].index[0]
        recommended_products = recommend_product(chosen_product_idx)

        # Display recommended products based on content-based filtering
        st.subheader("Recommended Products for You")
        recommended_products_list = recommended_products.to_dict('records')
        num_cols = 3
        cols = st.columns(num_cols)

        for i, product in enumerate(recommended_products_list):
            with cols[i % num_cols]:
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; height: 100%;">
                        <img src="{product['ImageURL']}" width="100%" style="border-radius: 8px; object-fit: contain; max-height: 200px;" />
                        <div style="flex-grow: 1;">
                            <h5>{format_name(product['Name'])}</h5>
                            <p><strong>Price:</strong> ${product['Price']}</p>
                            <p><strong>Rating:</strong> {product['Rating']} <strong>Reviews:</strong> {product['ReviewCount']}</p>
                        </div>
                        <p><a href="{product['URL']}" target="_blank" style="text-decoration: none; color: white; background-color: #4CAF50; padding: 8px; border-radius: 4px; display: block; text-align: center;">Buy Now</a></p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No products found matching your search.")    

# Always show top products based on weighted recommender
top_rated_products = weighted_recommender(df)
st.subheader("Top Products")

top_rated_products_list = top_rated_products.to_dict('records')
num_cols = 3
cols = st.columns(num_cols)

for i, product in enumerate(top_rated_products_list):
    with cols[i % num_cols]:
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; height: 100%;">
                <img src="{product['ImageURL']}" width="100%" style="border-radius: 8px; object-fit: contain; max-height: 200px;" />
                <div style="flex-grow: 1;">
                    <h5>{format_name(product['Name'])}</h5>
                    <p><strong>Price:</strong> ${product['Price']}</p>
                    <p><strong>Rating:</strong> {product['Rating']}<br><strong>Reviews:</strong> {product['ReviewCount']}</p>
                </div>
                <p><a href="{product['URL']}" target="_blank" style="text-decoration: none; color: white; background-color: #4CAF50; padding: 8px; border-radius: 4px; display: block; text-align: center;">Buy Now</a></p>
            </div>
        """, unsafe_allow_html=True)