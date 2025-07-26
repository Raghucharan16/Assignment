"""
Main script for Product Recommendation System
Hirebie AI ML Internship Assignment
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import ProductDataGenerator
from src.content_based_recommender import ContentBasedRecommender
from src.collaborative_recommender import CollaborativeRecommender

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "models/trained_models",
        "logs",
        "plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run the Product Recommendation System"""
    logger = setup_logging()
    
    logger.info("Starting Product Recommendation System")
    logger.info("=" * 50)
    
    # Create necessary directories
    create_directories()
    logger.info("Directories created successfully")
    
    try:
        # Step 1: Generate Product Dataset
        logger.info("Step 1: Generating product dataset...")
        data_generator = ProductDataGenerator()
        products_df, ratings_df = data_generator.generate_datasets()
        
        # Save datasets
        products_df.to_csv('data/raw/products.csv', index=False)
        ratings_df.to_csv('data/raw/ratings.csv', index=False)
        logger.info("Product datasets created and saved")
        
        # Step 2: Content-Based Recommendations
        logger.info("Step 2: Building content-based recommender...")
        content_recommender = ContentBasedRecommender()
        content_recommender.fit(products_df)
        content_recommender.save_model('models/content_based_model.pkl')
        logger.info("Content-based recommender trained and saved")
        
        # Step 3: Collaborative Filtering (Optional)
        logger.info("Step 3: Building collaborative filtering recommender...")
        collab_recommender = CollaborativeRecommender()
        collab_recommender.fit(ratings_df, products_df)
        collab_recommender.save_model('models/collaborative_model.pkl')
        logger.info("Collaborative filtering recommender trained and saved")
        
        # Step 4: Test Recommendations
        logger.info("Step 4: Testing recommendation system...")
        
        # Test content-based recommendations
        sample_product_id = products_df.iloc[0]['product_id']
        content_recs = content_recommender.get_recommendations(sample_product_id, top_k=5)
        logger.info(f"Content-based recommendations for product {sample_product_id}: {content_recs}")
        
        # Test collaborative filtering recommendations
        sample_user_id = ratings_df.iloc[0]['user_id']
        collab_recs = collab_recommender.get_recommendations(sample_user_id, top_k=5)
        logger.info(f"Collaborative filtering recommendations for user {sample_user_id}: {collab_recs}")
        
        # Step 5: Summary
        logger.info("Step 5: System summary...")
        logger.info("Product Recommendation System completed successfully!")
        logger.info("Check the following outputs:")
        logger.info("   - data/raw/ - Product and rating datasets")
        logger.info("   - models/ - Trained recommendation models")
        logger.info("   - logs/ - Log files")
        
        print("\n" + "="*60)
        print("PRODUCT RECOMMENDATION SYSTEM READY!")
        print("="*60)
        print("Next Steps:")
        print("1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("2. Explore Jupyter notebooks for analysis")
        print("3. Check model performance in logs/")
        print("4. Test recommendations through the web interface")
        print("="*60)
        
    except Exception as e:
        logger.error(f"System setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
