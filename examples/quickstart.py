"""
Quickstart Example for SOTA Recommender Library.

This example demonstrates basic usage of different recommender algorithms.
"""
import sys
sys.path.insert(0, '..')

import recommender
from recommender import (
    EASERecommender,
    SLIMRecommender,
    SVDRecommender,
    ALSRecommender,
    create_synthetic_dataset,
    InteractionDataset,
    Evaluator
)

def main():
    print("=" * 60)
    print("SOTA Recommender Library - Quickstart Example")
    print("=" * 60)
    
    # 1. Load or create dataset
    print("\n1. Loading dataset...")
    df = create_synthetic_dataset(
        n_users=1000,
        n_items=500,
        n_interactions=10000,
        implicit=True,
        seed=42
    )
    
    # 2. Create InteractionDataset
    print("\n2. Creating dataset...")
    dataset = InteractionDataset(df, implicit=True, min_user_interactions=5)
    
    # 3. Split data
    print("\n3. Splitting data...")
    train_data, test_data = dataset.split(test_size=0.2, strategy='random')
    print(f"Train: {len(train_data)} interactions")
    print(f"Test: {len(test_data)} interactions")
    
    # 4. Initialize evaluator
    evaluator = Evaluator(
        metrics=['precision', 'recall', 'ndcg', 'hit_rate'],
        k_values=[5, 10, 20]
    )
    
    # 5. Test different models
    models = {
        'EASE': EASERecommender(l2_reg=500.0),
        'SLIM': SLIMRecommender(l1_reg=0.1, l2_reg=0.1, max_iter=50),
        'SVD': SVDRecommender(n_components=20),
        'ALS': ALSRecommender(n_factors=20, n_iterations=10)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        try:
            # Train
            model.fit(train_data.data)
            
            # Evaluate
            print(f"\nEvaluating {model_name}...")
            model_results = evaluator.evaluate(
                model,
                test_data,
                task='ranking',
                exclude_train=True,
                train_data=train_data
            )
            
            results[model_name] = model_results
            evaluator.print_results(model_results)
            
            # Generate sample recommendations
            print(f"\nSample recommendations from {model_name}:")
            sample_users = test_data.data['user_id'].unique()[:3]
            recommendations = model.recommend(sample_users, k=5, exclude_seen=True)
            
            for user_id in sample_users:
                recs = recommendations.get(user_id, [])
                print(f"  User {user_id}: {[item_id for item_id, _ in recs[:5]]}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Compare results
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    # Compare NDCG@10
    print("\nNDCG@10:")
    for model_name in results:
        ndcg = results[model_name].get('ndcg@10', 0.0)
        print(f"  {model_name:15s}: {ndcg:.4f}")
    
    print("\nRecall@10:")
    for model_name in results:
        recall = results[model_name].get('recall@10', 0.0)
        print(f"  {model_name:15s}: {recall:.4f}")
    
    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

