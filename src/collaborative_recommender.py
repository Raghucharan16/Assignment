"""
Collaborative Filtering Recommendation System
Uses Matrix Factorization (NMF) for user-based recommendations
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import pickle
import logging

class CollaborativeRecommender:
    """Collaborative filtering recommender using NMF matrix factorization"""
    
    def __init__(self, n_factors=50, max_iter=200, alpha=0.1, random_state=42):
        """
        Initialize collaborative filtering recommender
        
        Args:
            n_factors: Number of factors for matrix factorization
            max_iter: Maximum number of iterations
            alpha: Regularization parameter
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        
        self.model = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
        self.ratings_df = None
        self.products_df = None
        self.user_mean_ratings = {}
        self.product_mean_ratings = {}
        self.global_mean = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, ratings_df, products_df=None):
        """
        Train the collaborative filtering model
        
        Args:
            ratings_df: DataFrame with user ratings (user_id, product_id, rating)
            products_df: Optional product information for enhanced recommendations
        """
        self.logger.info("Training collaborative filtering model...")
        
        try:
            self.ratings_df = ratings_df.copy()
            self.products_df = products_df
            
            # Create user-item matrix
            self._create_user_item_matrix(ratings_df)
            
            # Calculate statistics
            self._calculate_statistics(ratings_df)
            
            # Train NMF model
            self._train_model()
            
            self.logger.info("Collaborative filtering model trained successfully!")
            return self
            
        except Exception as e:
            self.logger.error(f"Error training collaborative model: {str(e)}")
            raise
    
    def _create_user_item_matrix(self, ratings_df):
        """Create user-item interaction matrix"""
        # Get unique users and items
        unique_users = sorted(ratings_df['user_id'].unique())
        unique_items = sorted(ratings_df['product_id'].unique())
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # Fill matrix with ratings
        for _, row in ratings_df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['product_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
    
    def _calculate_statistics(self, ratings_df):
        """Calculate various rating statistics"""
        self.global_mean = ratings_df['rating'].mean()
        
        # User mean ratings
        user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count'])
        self.user_mean_ratings = user_stats['mean'].to_dict()
        
        # Product mean ratings
        product_stats = ratings_df.groupby('product_id')['rating'].agg(['mean', 'count'])
        self.product_mean_ratings = product_stats['mean'].to_dict()
    
    def _train_model(self):
        """Train the NMF model"""
        # Use only non-zero entries for training
        mask = self.user_item_matrix > 0
        
        # Initialize NMF model (removed alpha parameter for compatibility)
        self.model = NMF(
            n_components=self.n_factors,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init='random'
        )
        
        # Fit the model
        self.W = self.model.fit_transform(self.user_item_matrix)
        self.H = self.model.components_
        
        # Calculate reconstruction error
        reconstructed = np.dot(self.W, self.H)
        mse = mean_squared_error(
            self.user_item_matrix[mask], 
            reconstructed[mask]
        )
        self.logger.info(f"Training MSE: {mse:.4f}")
    
    def predict_rating(self, user_id, product_id):
        """
        Predict rating for a specific user-product pair
        
        Args:
            user_id: User identifier
            product_id: Product identifier
            
        Returns:
            Predicted rating (float)
        """
        try:
            # Handle cold start - unknown user or item
            if user_id not in self.user_to_idx:
                return self.product_mean_ratings.get(product_id, self.global_mean)
            
            if product_id not in self.item_to_idx:
                return self.user_mean_ratings.get(user_id, self.global_mean)
            
            # Get indices
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[product_id]
            
            # Predict rating
            prediction = np.dot(self.W[user_idx], self.H[:, item_idx])
            
            # Clip to valid rating range
            return np.clip(prediction, 1.0, 5.0)
            
        except Exception:
            return self.global_mean
    
    def get_recommendations(self, user_id, top_k=5, exclude_rated=True):
        """
        Get product recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations to return
            exclude_rated: Whether to exclude already rated products
            
        Returns:
            List of recommended products with predicted ratings
        """
        try:
            # Check if model is properly initialized
            if not hasattr(self, 'W') or not hasattr(self, 'H') or self.W is None or self.H is None:
                self.logger.error("Model not properly initialized")
                return self._get_popular_products(top_k)
            
            if user_id not in self.user_to_idx:
                # Cold start: return popular products
                return self._get_popular_products(top_k)
            
            user_idx = self.user_to_idx[user_id]
            
            # Get predictions for all items
            user_vector = self.W[user_idx]
            predictions = np.dot(user_vector, self.H)
            
            # Create list of (item_idx, prediction) pairs
            item_predictions = []
            for item_idx, pred in enumerate(predictions):
                product_id = self.idx_to_item[item_idx]
                
                # Skip if already rated and exclude_rated is True
                if exclude_rated and hasattr(self, 'user_item_matrix') and self.user_item_matrix is not None:
                    if self.user_item_matrix[user_idx, item_idx] > 0:
                        continue
                
                # Clip prediction to valid range
                pred_clipped = np.clip(pred, 1.0, 5.0)
                item_predictions.append((product_id, pred_clipped))
            
            # Sort by prediction score and return top N
            item_predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = item_predictions[:top_k]
            
            # Enhance with product information if available
            recommendations = []
            for product_id, predicted_rating in top_predictions:
                rec = {
                    'product_id': product_id,
                    'predicted_rating': round(predicted_rating, 2)
                }
                
                # Add product details if available
                if self.products_df is not None:
                    product_info = self.products_df[
                        self.products_df['product_id'] == product_id
                    ]
                    if not product_info.empty:
                        product_info = product_info.iloc[0]
                        rec.update({
                            'title': product_info['title'],
                            'category': product_info['category'],
                            'brand': product_info['brand'],
                            'price': product_info['price'],
                            'rating': product_info['rating'],
                            'description': product_info['description']
                        })
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
            return self._get_popular_products(top_k)
    
    def _get_popular_products(self, top_k):
        """Get popular products for cold start scenarios"""
        try:
            if self.products_df is not None and not self.products_df.empty:
                # Sort by rating and return top products
                popular = self.products_df.nlargest(top_k, 'rating')
                recommendations = []
                for _, row in popular.iterrows():
                    rec = {
                        'product_id': row['product_id'],
                        'predicted_rating': row['rating'],
                        'title': row['title'],
                        'category': row['category'],
                        'brand': row['brand'],
                        'price': row['price'],
                        'rating': row['rating'],
                        'description': row['description']
                    }
                    recommendations.append(rec)
                return recommendations
            elif hasattr(self, 'product_mean_ratings') and self.product_mean_ratings:
                # Fallback: use product mean ratings
                sorted_products = sorted(
                    self.product_mean_ratings.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                return [{'product_id': pid, 'predicted_rating': rating} 
                       for pid, rating in sorted_products[:top_k]]
            else:
                # Last resort: return empty list
                return []
        except Exception as e:
            self.logger.error(f"Error getting popular products: {str(e)}")
            return []
    
    def get_user_similarity(self, user_id1, user_id2):
        """Calculate similarity between two users based on their latent factors"""
        if user_id1 not in self.user_to_idx or user_id2 not in self.user_to_idx:
            return 0.0
        
        user1_idx = self.user_to_idx[user_id1]
        user2_idx = self.user_to_idx[user_id2]
        
        user1_vector = self.W[user1_idx]
        user2_vector = self.W[user2_idx]
        
        # Cosine similarity
        norm_product = np.linalg.norm(user1_vector) * np.linalg.norm(user2_vector)
        if norm_product > 0:
            similarity = np.dot(user1_vector, user2_vector) / norm_product
            return similarity
        return 0.0
    
    def get_similar_users(self, user_id, top_k=5):
        """Find users similar to given user"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.W[user_idx]
        
        # Calculate similarity with all users
        similarities = []
        for other_idx in range(self.W.shape[0]):
            if other_idx == user_idx:
                continue
            
            other_vector = self.W[other_idx]
            
            # Cosine similarity
            norm_product = np.linalg.norm(user_vector) * np.linalg.norm(other_vector)
            if norm_product > 0:
                similarity = np.dot(user_vector, other_vector) / norm_product
                other_user_id = self.idx_to_user[other_idx]
                similarities.append({
                    'user_id': other_user_id,
                    'similarity': similarity
                })
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_user_profile(self, user_id):
        """Get user's rating profile and preferences"""
        # Check if required data is available
        if self.ratings_df is None:
            return {
                'user_id': user_id,
                'total_ratings': 0,
                'average_rating': self.global_mean if hasattr(self, 'global_mean') and self.global_mean else 3.0,
                'status': 'Data Not Available'
            }
        
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if user_ratings.empty:
            return {
                'user_id': user_id,
                'total_ratings': 0,
                'average_rating': self.global_mean if hasattr(self, 'global_mean') and self.global_mean else 3.0,
                'status': 'New User'
            }
        
        profile = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'average_rating': user_ratings['rating'].mean(),
            'status': 'Active User' if len(user_ratings) >= 5 else 'Light User'
        }
        
        # Add rating distribution
        try:
            rating_dist = user_ratings['rating'].value_counts().sort_index()
            profile['rating_distribution'] = rating_dist.to_dict()
        except Exception:
            profile['rating_distribution'] = {}
        
        # Add category preferences if product data available
        try:
            if self.products_df is not None and not self.products_df.empty:
                user_products = user_ratings.merge(
                    self.products_df[['product_id', 'category']], 
                    on='product_id'
                )
                if not user_products.empty:
                    category_preferences = user_products.groupby('category')['rating'].mean().sort_values(ascending=False)
                    profile['category_preferences'] = category_preferences.to_dict()
        except Exception:
            # If category preferences can't be computed, skip it
            pass
        
        return profile
    
    def get_popular_products(self, top_k=10):
        """Get most popular products based on ratings"""
        product_stats = self.ratings_df.groupby('product_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        product_stats.columns = ['rating_count', 'avg_rating']
        product_stats = product_stats.reset_index()
        
        # Calculate popularity score (weighted rating)
        min_ratings = 3  # Minimum ratings required
        product_stats = product_stats[product_stats['rating_count'] >= min_ratings]
        
        if product_stats.empty:
            # If no products meet minimum threshold, use all products
            product_stats = self.ratings_df.groupby('product_id').agg({
                'rating': ['count', 'mean']
            }).round(2)
            product_stats.columns = ['rating_count', 'avg_rating']
            product_stats = product_stats.reset_index()
        
        # Sort by average rating and rating count
        product_stats['popularity_score'] = (
            product_stats['avg_rating'] * 0.7 + 
            (product_stats['rating_count'] / product_stats['rating_count'].max()) * 0.3
        )
        
        popular_products = product_stats.nlargest(top_k, 'popularity_score')
        
        # Enhance with product details
        recommendations = []
        for _, product in popular_products.iterrows():
            rec = {
                'product_id': product['product_id'],
                'avg_rating': product['avg_rating'],
                'rating_count': int(product['rating_count']),
                'popularity_score': round(product['popularity_score'], 3)
            }
            
            if self.products_df is not None:
                product_info = self.products_df[
                    self.products_df['product_id'] == product['product_id']
                ]
                if not product_info.empty:
                    product_info = product_info.iloc[0]
                    rec.update({
                        'title': product_info['title'],
                        'category': product_info['category'],
                        'brand': product_info['brand'],
                        'price': product_info['price'],
                        'description': product_info['description']
                    })
            
            recommendations.append(rec)
        
        return recommendations
    
    def save_model(self, filepath):
        """Save the trained model as complete object"""
        # Save the complete object
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Collaborative filtering model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # If it's a dictionary (old format), create new instance and load data
        if isinstance(model, dict):
            new_model = cls()
            new_model.W = model.get('W')
            new_model.H = model.get('H') 
            new_model.user_to_idx = model.get('user_to_idx', {})
            new_model.item_to_idx = model.get('item_to_idx', {})
            new_model.idx_to_user = model.get('idx_to_user', {})
            new_model.idx_to_item = model.get('idx_to_item', {})
            new_model.user_mean_ratings = model.get('user_mean_ratings', {})
            new_model.product_mean_ratings = model.get('product_mean_ratings', {})
            new_model.global_mean = model.get('global_mean', 3.0)
            new_model.n_factors = model.get('n_factors', 50)
            new_model.ratings_df = model.get('ratings_df')
            new_model.products_df = model.get('products_df')
            
            new_model.logger.info(f"Collaborative filtering model loaded from {filepath} (old format)")
            return new_model
        else:
            # New format - return the loaded object directly
            model.logger.info(f"Collaborative filtering model loaded from {filepath}")
            return model

if __name__ == "__main__":
    # Test the collaborative filtering recommender
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    from data_generator import ProductDataGenerator
    
    generator = ProductDataGenerator(n_products=20, n_users=50, n_ratings=200)
    products_df, ratings_df = generator.generate_datasets()
    
    # Train recommender
    recommender = CollaborativeRecommender()
    recommender.fit(ratings_df, products_df)
    
    # Test recommendations
    sample_user = ratings_df.iloc[0]['user_id']
    recommendations = recommender.get_recommendations(sample_user, top_k=5)
    
    print(f"\n🔍 Recommendations for user {sample_user}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.2f})")
    
    # Test user profile
    profile = recommender.get_user_profile(sample_user)
    print(f"\n👤 User Profile for {sample_user}:")
    print(f"   Total Ratings: {profile['total_ratings']}")
    print(f"   Average Rating: {profile['average_rating']:.2f}")
    # Test popular products
    popular = recommender.get_popular_products(top_k=3)
    print(f"\n🌟 Popular Products:")
    for i, product in enumerate(popular, 1):
        print(f"{i}. {product['title']} (Avg: {product['avg_rating']:.2f}, Count: {product['rating_count']})")
