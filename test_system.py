"""
Test script for Product Recommendation System
Verifies that all components work correctly
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_generation():
    """Test product data generation"""
    print("🧪 Testing Data Generation...")
    
    try:
        from src.data_generator import ProductDataGenerator
        
        generator = ProductDataGenerator(n_products=10, n_users=20, n_ratings=50)
        products_df, ratings_df = generator.generate_datasets()
        
        assert len(products_df) == 10, "Products dataset size mismatch"
        assert len(ratings_df) == 50, "Ratings dataset size mismatch"
        assert 'product_id' in products_df.columns, "Missing product_id column"
        assert 'title' in products_df.columns, "Missing title column"
        assert 'user_id' in ratings_df.columns, "Missing user_id column"
        
        print("   ✅ Data generation working correctly")
        return products_df, ratings_df
        
    except Exception as e:
        print(f"   ❌ Data generation failed: {str(e)}")
        return None, None

def test_content_based_recommender(products_df):
    """Test content-based recommender"""
    print("🧪 Testing Content-Based Recommender...")
    
    try:
        from src.content_based_recommender import ContentBasedRecommender
        
        recommender = ContentBasedRecommender()
        recommender.fit(products_df)
        
        # Test recommendations
        sample_product = products_df.iloc[0]['product_id']
        recommendations = recommender.get_recommendations(sample_product, top_k=3)
        
        assert len(recommendations) == 3, "Wrong number of recommendations"
        assert 'similarity_score' in recommendations[0], "Missing similarity score"
        assert recommendations[0]['similarity_score'] <= 1.0, "Invalid similarity score"
        
        print("   ✅ Content-based recommender working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Content-based recommender failed: {str(e)}")
        return False

def test_collaborative_recommender(ratings_df, products_df):
    """Test collaborative filtering recommender"""
    print("🧪 Testing Collaborative Filtering Recommender...")
    
    try:
        from src.collaborative_recommender import CollaborativeRecommender
        
        recommender = CollaborativeRecommender()
        recommender.fit(ratings_df, products_df)
        
        # Test recommendations
        sample_user = ratings_df.iloc[0]['user_id']
        recommendations = recommender.get_recommendations(sample_user, top_k=3)
        
        assert len(recommendations) <= 3, "Too many recommendations"
        assert 'predicted_rating' in recommendations[0], "Missing predicted rating"
        
        print("   ✅ Collaborative filtering recommender working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Collaborative filtering recommender failed: {str(e)}")
        return False

def test_system_integration():
    """Test full system integration"""
    print("🧪 Testing System Integration...")
    
    try:
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Generate data
        products_df, ratings_df = test_data_generation()
        if products_df is None:
            return False
        
        # Save data
        products_df.to_csv('data/raw/products.csv', index=False)
        ratings_df.to_csv('data/raw/ratings.csv', index=False)
        
        # Test content-based recommender
        content_success = test_content_based_recommender(products_df)
        
        # Test collaborative filtering
        collab_success = test_collaborative_recommender(ratings_df, products_df)
        
        if content_success and collab_success:
            print("   ✅ System integration successful")
            return True
        else:
            print("   ❌ System integration failed")
            return False
        
    except Exception as e:
        print(f"   ❌ System integration failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Product Recommendation System - Test Suite")
    print("=" * 60)
    
    success = test_system_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n📋 Next Steps:")
        print("1. Run: python main.py (to generate full dataset and train models)")
        print("2. Run: streamlit run streamlit_app.py (to start web interface)")
        print("3. Open: http://localhost:8501 (to use the application)")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\n🔧 Please check the error messages above and:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that all files are in the correct locations")
        print("3. Run the tests again")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
