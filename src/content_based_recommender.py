"""
Content-Based Recommendation System
Uses TF-IDF and cosine similarity for product recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

class ContentBasedRecommender:
    """Content-based product recommender using TF-IDF and cosine similarity"""
    
    def __init__(self):
        """Initialize the content-based recommender"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.products_df = None
        self.similarity_matrix = None
        self.product_to_index = {}
        self.index_to_product = {}
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, df):
        """Preprocess product text data for TF-IDF"""
        # Combine title, description, category, and brand into one text field
        df['combined_features'] = (
            df['title'].fillna('') + ' ' + 
            df['description'].fillna('') + ' ' + 
            df['category'].fillna('') + ' ' + 
            df['brand'].fillna('')
        )
        
        return df['combined_features']
    
    def fit(self, products_df):
        """
        Fit the content-based recommender
        
        Args:
            products_df: DataFrame with product information
        """
        self.logger.info("Training content-based recommender...")
        
        self.products_df = products_df.copy()
        
        # Preprocess text data
        combined_features = self.preprocess_text(self.products_df)
        
        # Create TF-IDF matrix
        self.logger.info("Creating TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_features)
        
        # Calculate cosine similarity matrix
        self.logger.info("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Create product ID to index mappings
        self.product_to_index = {
            product_id: idx for idx, product_id in enumerate(self.products_df['product_id'])
        }
        self.index_to_product = {
            idx: product_id for product_id, idx in self.product_to_index.items()
        }
        
        self.logger.info("Content-based recommender trained successfully!")
        
        # Log some statistics
        self.logger.info("Model Statistics:")
        self.logger.info(f"   - Products: {len(self.products_df)}")
        self.logger.info(f"   - TF-IDF features: {self.tfidf_matrix.shape[1]}")
        self.logger.info(f"   - Similarity matrix shape: {self.similarity_matrix.shape}")
    
    def get_recommendations(self, product_id, top_k=5, include_scores=True):
        """
        Get product recommendations based on content similarity
        
        Args:
            product_id: ID of the product to get recommendations for
            top_k: Number of recommendations to return
            include_scores: Whether to include similarity scores
            
        Returns:
            List of recommended products with optional similarity scores
        """
        if product_id not in self.product_to_index:
            self.logger.error(f"Product ID {product_id} not found in dataset")
            return []
        
        # Get product index
        product_idx = self.product_to_index[product_id]
        
        # Get similarity scores for this product
        similarity_scores = self.similarity_matrix[product_idx]
        
        # Get indices of most similar products (excluding the product itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_k+1]
        
        recommendations = []
        for idx in similar_indices:
            rec_product_id = self.index_to_product[idx]
            product_info = self.products_df[
                self.products_df['product_id'] == rec_product_id
            ].iloc[0]
            
            rec = {
                'product_id': rec_product_id,
                'title': product_info['title'],
                'category': product_info['category'],
                'brand': product_info['brand'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'description': product_info['description']
            }
            
            if include_scores:
                rec['similarity_score'] = round(similarity_scores[idx], 4)
            
            recommendations.append(rec)
        
        return recommendations
    
    def get_recommendations_by_category(self, category, top_k=10):
        """Get top products from a specific category"""
        category_products = self.products_df[
            self.products_df['category'] == category
        ].sort_values('rating', ascending=False).head(top_k)
        
        recommendations = []
        for _, product in category_products.iterrows():
            rec = {
                'product_id': product['product_id'],
                'title': product['title'],
                'category': product['category'],
                'brand': product['brand'],
                'price': product['price'],
                'rating': product['rating'],
                'description': product['description']
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_similar_by_text(self, search_text, top_k=5):
        """Get products similar to given text description"""
        # Transform search text using fitted TF-IDF vectorizer
        search_tfidf = self.tfidf_vectorizer.transform([search_text])
        
        # Calculate similarity with all products
        search_similarity = cosine_similarity(search_tfidf, self.tfidf_matrix)[0]
        
        # Get top similar products
        similar_indices = search_similarity.argsort()[::-1][:top_k]
        
        recommendations = []
        for idx in similar_indices:
            product_id = self.index_to_product[idx]
            product_info = self.products_df[
                self.products_df['product_id'] == product_id
            ].iloc[0]
            
            rec = {
                'product_id': product_id,
                'title': product_info['title'],
                'category': product_info['category'],
                'brand': product_info['brand'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'description': product_info['description'],
                'similarity_score': round(search_similarity[idx], 4)
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def get_product_info(self, product_id):
        """Get detailed information about a specific product"""
        if product_id not in self.product_to_index:
            return None
        
        product_info = self.products_df[
            self.products_df['product_id'] == product_id
        ].iloc[0]
        
        return {
            'product_id': product_info['product_id'],
            'title': product_info['title'],
            'description': product_info['description'],
            'category': product_info['category'],
            'brand': product_info['brand'],
            'price': product_info['price'],
            'rating': product_info['rating']
        }
    
    def get_all_products(self):
        """Get list of all products for UI dropdown"""
        return self.products_df[['product_id', 'title', 'category', 'price']].to_dict('records')
    
    def get_categories(self):
        """Get list of all categories"""
        return sorted(self.products_df['category'].unique().tolist())
    
    def get_feature_importance(self, product_id, top_features=10):
        """Get most important TF-IDF features for a product"""
        if product_id not in self.product_to_index:
            return []
        
        product_idx = self.product_to_index[product_id]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for this product
        tfidf_scores = self.tfidf_matrix[product_idx].toarray()[0]
        
        # Get top features
        top_indices = tfidf_scores.argsort()[::-1][:top_features]
        
        important_features = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0:
                important_features.append({
                    'feature': feature_names[idx],
                    'score': round(tfidf_scores[idx], 4)
                })
        
        return important_features
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'similarity_matrix': self.similarity_matrix,
            'products_df': self.products_df,
            'product_to_index': self.product_to_index,
            'index_to_product': self.index_to_product
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"💾 Content-based model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.similarity_matrix = model_data['similarity_matrix']
        self.products_df = model_data['products_df']
        self.product_to_index = model_data['product_to_index']
        self.index_to_product = model_data['index_to_product']
        
        self.logger.info(f"📁 Model loaded from {filepath}")

if __name__ == "__main__":
    # Test the content-based recommender
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data (you would replace this with actual data loading)
    from data_generator import ProductDataGenerator
    
    generator = ProductDataGenerator(n_products=20)
    products_df, _ = generator.generate_datasets()
    
    # Train recommender
    recommender = ContentBasedRecommender()
    recommender.fit(products_df)
    
    # Test recommendations
    sample_product = products_df.iloc[0]['product_id']
    recommendations = recommender.get_recommendations(sample_product, top_k=5)
    
    print(f"\n🔍 Recommendations for product {sample_product}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
    
    # Test feature importance
    features = recommender.get_feature_importance(sample_product, top_features=5)
    print(f"\n📊 Important features for {sample_product}:")
    for feature in features:
        print(f"   {feature['feature']}: {feature['score']:.3f}")
