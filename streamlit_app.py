"""
Streamlit Web Application for Product Recommendation System
Hirebie AI ML Internship Assignment
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging
import warnings
from datetime import datetime

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_generator import ProductDataGenerator
    from src.content_based_recommender import ContentBasedRecommender
    from src.collaborative_recommender import CollaborativeRecommender
except ImportError:
    st.error("Please run 'python main.py' first to generate data and train models.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load product and rating datasets"""
    try:
        products_df = pd.read_csv('data/raw/products.csv')
        ratings_df = pd.read_csv('data/raw/ratings.csv')
        return products_df, ratings_df
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_models():
    """Load trained recommendation models"""
    content_model = ContentBasedRecommender()
    collab_model = CollaborativeRecommender()
    
    try:
        # Load content-based model
        if os.path.exists('models/content_based_model.pkl'):
            content_model.load_model('models/content_based_model.pkl')
        else:
            st.warning("Content-based model not found. Please run main.py first.")
            return None, None
        
        # Load collaborative filtering model
        if os.path.exists('models/collaborative_model.pkl'):
            collab_model = CollaborativeRecommender.load_model('models/collaborative_model.pkl')
        else:
            st.warning("Collaborative filtering model not found. Please run main.py first.")
            return content_model, None
            
        return content_model, collab_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def display_product_card(product, similarity_score=None, predicted_rating=None):
    """Display a product recommendation card"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{product['title']}**")
            st.write(f"🏷️ {product['category']} | 🏢 {product['brand']}")
            st.write(f"💰 ${product['price']} | ⭐ {product['rating']}/5.0")
            with st.expander("Description"):
                st.write(product['description'])
        
        with col2:
            if similarity_score is not None:
                st.metric("Similarity", f"{similarity_score:.3f}")
            if predicted_rating is not None:
                st.metric("Predicted Rating", f"{predicted_rating:.2f}/5.0")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🛍️ Product Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("**Hirebie AI ML Internship Assignment**")
    st.markdown("---")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        products_df, ratings_df = load_data()
        content_model, collab_model = load_models()
    
    if products_df is None or content_model is None:
        st.error("🚨 Data or models not found. Please run the following commands:")
        st.code("python main.py")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🎛️ Controls</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rec_type' not in st.session_state:
        st.session_state.rec_type = "Content-Based Filtering"
    
    # Recommendation type selection
    rec_type = st.sidebar.selectbox(
        "Choose Recommendation Type",
        ["Content-Based Filtering", "Collaborative Filtering", "Popular Products", "Category Browse"],
        key="recommendation_type_selector"
    )
    
    # Main content area
    if rec_type == "Content-Based Filtering":
        st.header("🔍 Content-Based Recommendations")
        st.write("Get recommendations based on product features and descriptions.")
        
        # Product selection
        try:
            all_products = content_model.get_all_products()
            product_options = {f"{p['title']} ({p['category']})": p['product_id'] 
                              for p in all_products}
            
            selected_product_display = st.selectbox(
                "Select a product to get similar recommendations:",
                list(product_options.keys()),
                key="content_product_selector"
            )
        except Exception as e:
            st.error(f"Error loading products: {str(e)}")
            return
        
        if selected_product_display:
            selected_product_id = product_options[selected_product_display]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                num_recommendations = st.slider("Number of recommendations", 1, 10, 5, key="content_num_recs")
            
            with col2:
                show_scores = st.checkbox("Show similarity scores", value=True, key="content_show_scores")
            
            if st.button("Get Recommendations", type="primary", key="content_get_recs"):
                with st.spinner("Finding similar products..."):
                    try:
                        # Get selected product info
                        selected_product = content_model.get_product_info(selected_product_id)
                        
                        # Display selected product
                        st.subheader("📦 Selected Product")
                        with st.container():
                            st.markdown(f"**{selected_product['title']}**")
                            st.write(f"🏷️ {selected_product['category']} | 🏢 {selected_product['brand']}")
                            st.write(f"💰 ${selected_product['price']} | ⭐ {selected_product['rating']}/5.0")
                            st.write(selected_product['description'])
                        
                        # Get recommendations
                        recommendations = content_model.get_recommendations(
                            selected_product_id, 
                            top_k=num_recommendations,
                            include_scores=show_scores
                        )
                        
                        st.subheader(f"🎯 Top {num_recommendations} Similar Products")
                        
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"### {i}. Recommendation")
                            similarity_score = rec.get('similarity_score') if show_scores else None
                            display_product_card(rec, similarity_score=similarity_score)
                            st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
                        st.write("Please make sure the models are properly trained.")
    
    elif rec_type == "Collaborative Filtering":
        if collab_model is None:
            st.error("Collaborative filtering model not available.")
            return
            
        st.header("👥 Collaborative Filtering Recommendations")
        st.write("Get recommendations based on user preferences and ratings.")
        
        # User selection
        all_users = sorted(ratings_df['user_id'].unique())
        selected_user = st.selectbox("Select a user:", all_users, key="collab_user_selector")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_recommendations = st.slider("Number of recommendations", 1, 10, 5, key="collab_num_recs")
        
        with col2:
            show_ratings = st.checkbox("Show predicted ratings", value=True, key="collab_show_ratings")
        
        if st.button("Get User Recommendations", type="primary", key="collab_get_recs"):
            with st.spinner("Analyzing user preferences..."):
                try:
                    # Get user profile
                    user_profile = collab_model.get_user_profile(selected_user)
                    
                    # Check if user profile is valid
                    if user_profile is None:
                        st.error("Unable to get user profile. Please check if the model is properly trained.")
                        return
                    
                    # Display user profile
                    st.subheader(f"👤 User Profile: {selected_user}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Ratings", user_profile.get('total_ratings', 0))
                    
                    with col2:
                        avg_rating = user_profile.get('average_rating', 0)
                        st.metric("Average Rating", f"{avg_rating:.2f}" if avg_rating else "N/A")
                    
                    with col3:
                        if 'category_preferences' in user_profile and user_profile['category_preferences']:
                            try:
                                fav_category = max(user_profile['category_preferences'], 
                                                 key=user_profile['category_preferences'].get)
                                st.metric("Favorite Category", fav_category)
                            except:
                                st.metric("Favorite Category", "N/A")
                        else:
                            st.metric("Favorite Category", "N/A")
                    
                    # Get recommendations
                    recommendations = collab_model.get_recommendations(
                        selected_user, 
                        top_k=num_recommendations
                    )
                    
                    if not recommendations:
                        st.warning("No recommendations available for this user.")
                        return
                    
                    st.subheader(f"🎯 Top {len(recommendations)} Recommended Products")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"### {i}. Recommendation")
                        predicted_rating = rec.get('predicted_rating') if show_ratings else None
                        display_product_card(rec, predicted_rating=predicted_rating)
                        st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error getting user recommendations: {str(e)}")
                    st.write("Please make sure the collaborative filtering model is properly trained.")
                    st.write("You can try:")
                    st.code("python main.py")
                    st.write("to retrain the models.")
    
    elif rec_type == "Popular Products":
        st.header("🌟 Popular Products")
        st.write("Discover the most popular products based on user ratings.")
        
        num_products = st.slider("Number of products to show", 5, 20, 10, key="popular_num_products")
        
        if collab_model:
            try:
                popular_products = collab_model.get_popular_products(top_k=num_products)
                
                st.subheader(f"🏆 Top {num_products} Popular Products")
                
                for i, product in enumerate(popular_products, 1):
                    st.markdown(f"### {i}. Popular Product")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{product['title']}**")
                        st.write(f"🏷️ {product['category']} | 🏢 {product['brand']}")
                        st.write(f"💰 ${product['price']}")
                        with st.expander("Description"):
                            st.write(product['description'])
                    
                    with col2:
                        st.metric("Avg Rating", f"{product['avg_rating']:.2f}/5.0")
                        st.metric("Total Ratings", product['rating_count'])
                        st.metric("Popularity Score", f"{product['popularity_score']:.3f}")
                    
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error getting popular products: {str(e)}")
        else:
            st.error("Collaborative filtering model required for popularity analysis.")
    
    elif rec_type == "Category Browse":
        st.header("🗂️ Browse by Category")
        st.write("Explore products by category.")
        
        try:
            # Category selection
            categories = content_model.get_categories()
            selected_category = st.selectbox("Select a category:", categories, key="category_selector")
            
            num_products = st.slider("Number of products to show", 5, 20, 10, key="category_num_products")
            
            if st.button("Browse Category", type="primary", key="category_browse"):
                category_products = content_model.get_recommendations_by_category(
                    selected_category, 
                    top_k=num_products
                )
                
                st.subheader(f"🏷️ {selected_category} Products")
                
                for i, product in enumerate(category_products, 1):
                    st.markdown(f"### {i}. {product['title']}")
                    display_product_card(product)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error browsing category: {str(e)}")
    
    # Sidebar statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Dataset Statistics**")
    st.sidebar.metric("Total Products", len(products_df))
    st.sidebar.metric("Categories", products_df['category'].nunique())
    st.sidebar.metric("Brands", products_df['brand'].nunique())
    
    if ratings_df is not None:
        st.sidebar.metric("Total Ratings", len(ratings_df))
        st.sidebar.metric("Total Users", ratings_df['user_id'].nunique())
        st.sidebar.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")
    

if __name__ == "__main__":
    main()
