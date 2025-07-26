"""
Product Data Generator for Recommendation System
Creates realistic product and rating datasets
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

class ProductDataGenerator:
    """Generate realistic product and user rating datasets"""
    
    def __init__(self, n_products=50, n_users=100, n_ratings=500):
        """
        Initialize data generator
        
        Args:
            n_products: Number of products to generate
            n_users: Number of users for ratings
            n_ratings: Number of ratings to generate
        """
        self.n_products = n_products
        self.n_users = n_users
        self.n_ratings = n_ratings
        
        # Product categories and their typical products
        self.categories = {
            'Electronics': {
                'products': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smart Watch', 'Camera', 'Speaker', 'Monitor'],
                'brands': ['Apple', 'Samsung', 'Sony', 'LG', 'HP', 'Dell', 'Canon', 'Bose'],
                'price_range': (50, 2000)
            },
            'Clothing': {
                'products': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Sweater', 'Pants', 'Shirt'],
                'brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Levi\'s', 'Gap', 'Uniqlo', 'Puma'],
                'price_range': (20, 300)
            },
            'Home & Kitchen': {
                'products': ['Coffee Maker', 'Blender', 'Microwave', 'Vacuum', 'Air Fryer', 'Dishware Set', 'Cookware', 'Furniture'],
                'brands': ['KitchenAid', 'Cuisinart', 'Dyson', 'Ninja', 'IKEA', 'Hamilton Beach', 'Instant Pot', 'Breville'],
                'price_range': (25, 800)
            },
            'Books': {
                'products': ['Fiction Novel', 'Non-Fiction', 'Textbook', 'Biography', 'Self-Help', 'Cookbook', 'Travel Guide', 'Art Book'],
                'brands': ['Penguin', 'Random House', 'Springer', 'O\'Reilly', 'National Geographic', 'Lonely Planet', 'McGraw Hill', 'Wiley'],
                'price_range': (10, 150)
            },
            'Sports & Outdoors': {
                'products': ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Bicycle', 'Backpack', 'Tent', 'Water Bottle', 'Fitness Tracker'],
                'brands': ['Nike', 'Adidas', 'Under Armour', 'Patagonia', 'The North Face', 'Coleman', 'Hydro Flask', 'Fitbit'],
                'price_range': (15, 1200)
            }
        }
        
        np.random.seed(42)
        random.seed(42)
    
    def generate_product_descriptions(self, product_name, category, brand):
        """Generate realistic product descriptions"""
        
        description_templates = {
            'Electronics': [
                f"High-quality {product_name} from {brand} featuring advanced technology and sleek design.",
                f"Professional-grade {product_name} with cutting-edge features and excellent performance.",
                f"Premium {product_name} offering superior quality and innovative functionality.",
                f"{brand} {product_name} with state-of-the-art technology and user-friendly interface."
            ],
            'Clothing': [
                f"Stylish and comfortable {product_name} from {brand}, perfect for everyday wear.",
                f"High-quality {product_name} made with premium materials and modern design.",
                f"Trendy {product_name} by {brand} combining style, comfort, and durability.",
                f"Fashion-forward {product_name} featuring contemporary design and excellent fit."
            ],
            'Home & Kitchen': [
                f"Essential {product_name} from {brand} designed to make your daily tasks easier.",
                f"Professional-quality {product_name} with durable construction and reliable performance.",
                f"Innovative {product_name} featuring modern design and practical functionality.",
                f"{brand} {product_name} combining efficiency, style, and user convenience."
            ],
            'Books': [
                f"Engaging {product_name} from {brand} offering valuable insights and knowledge.",
                f"Well-researched {product_name} providing comprehensive information and expert guidance.",
                f"Informative {product_name} featuring clear explanations and practical examples.",
                f"Authoritative {product_name} by leading experts in the field."
            ],
            'Sports & Outdoors': [
                f"High-performance {product_name} from {brand} designed for active lifestyles.",
                f"Durable {product_name} built to withstand challenging conditions and intensive use.",
                f"Professional-grade {product_name} offering superior performance and reliability.",
                f"{brand} {product_name} engineered for optimal comfort and functionality."
            ]
        }
        
        return random.choice(description_templates[category])
    
    def generate_products_dataset(self):
        """Generate products dataset"""
        products = []
        
        for i in range(self.n_products):
            # Select random category
            category = random.choice(list(self.categories.keys()))
            category_info = self.categories[category]
            
            # Select product type and brand
            product_type = random.choice(category_info['products'])
            brand = random.choice(category_info['brands'])
            
            # Generate price
            min_price, max_price = category_info['price_range']
            price = round(np.random.uniform(min_price, max_price), 2)
            
            # Generate product name
            product_name = f"{brand} {product_type}"
            if random.random() > 0.7:  # 30% chance to add model/version
                model = random.choice(['Pro', 'Max', 'Ultra', 'Plus', 'Elite', 'Premium', 'Deluxe'])
                product_name += f" {model}"
            
            # Generate description
            description = self.generate_product_descriptions(product_type, category, brand)
            
            # Generate rating (4.0-5.0 with normal distribution)
            rating = round(np.clip(np.random.normal(4.2, 0.4), 3.0, 5.0), 1)
            
            product = {
                'product_id': f'P{i+1:03d}',
                'title': product_name,
                'description': description,
                'category': category,
                'brand': brand,
                'price': price,
                'rating': rating,
                'created_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            }
            
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_ratings_dataset(self, products_df):
        """Generate user ratings dataset"""
        ratings = []
        
        for i in range(self.n_ratings):
            user_id = f'U{random.randint(1, self.n_users):03d}'
            product_id = random.choice(products_df['product_id'].tolist())
            
            # Get product info to influence rating
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            base_rating = product_info['rating']
            
            # Generate user rating around product's average rating
            user_rating = np.clip(np.random.normal(base_rating, 0.5), 1.0, 5.0)
            user_rating = round(user_rating * 2) / 2  # Round to nearest 0.5
            
            # Generate timestamp
            rating_date = datetime.now() - timedelta(days=random.randint(1, 180))
            
            rating = {
                'user_id': user_id,
                'product_id': product_id,
                'rating': user_rating,
                'timestamp': rating_date.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            ratings.append(rating)
        
        return pd.DataFrame(ratings)
    
    def generate_datasets(self):
        """Generate both products and ratings datasets"""
        print("🏭 Generating product dataset...")
        products_df = self.generate_products_dataset()
        
        print("⭐ Generating ratings dataset...")
        ratings_df = self.generate_ratings_dataset(products_df)
        
        print(f"✅ Generated {len(products_df)} products and {len(ratings_df)} ratings")
        
        return products_df, ratings_df
    
    def save_datasets(self, products_df, ratings_df, output_dir='data/raw/'):
        """Save datasets to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        ratings_df.to_csv(f'{output_dir}/ratings.csv', index=False)
        
        # Save as JSON for reference
        products_df.to_json(f'{output_dir}/products.json', orient='records', indent=2)
        ratings_df.to_json(f'{output_dir}/ratings.json', orient='records', indent=2)
        
        print(f"💾 Datasets saved to {output_dir}")
        
        # Print summary statistics
        print("\n📊 Dataset Summary:")
        print(f"Products: {len(products_df)}")
        print(f"Categories: {products_df['category'].nunique()}")
        print(f"Brands: {products_df['brand'].nunique()}")
        print(f"Price range: ${products_df['price'].min():.2f} - ${products_df['price'].max():.2f}")
        print(f"Ratings: {len(ratings_df)}")
        print(f"Users: {ratings_df['user_id'].nunique()}")
        print(f"Average rating: {ratings_df['rating'].mean():.2f}")

if __name__ == "__main__":
    # Generate and save datasets
    generator = ProductDataGenerator(n_products=50, n_users=100, n_ratings=500)
    products_df, ratings_df = generator.generate_datasets()
    generator.save_datasets(products_df, ratings_df)
    
    # Display sample data
    print("\n🔍 Sample Products:")
    print(products_df.head())
    
    print("\n⭐ Sample Ratings:")
    print(ratings_df.head())
