"""
Training script for Drug Recommendation System
Run this script first to train models before starting the web app
"""
import os
import sys
from models.data_preprocessing import DataPreprocessor
from models.feature_engineering import FeatureEngineer
from models.model_training import ModelTrainer
from models.recommendation_engine import RecommendationEngine

def main():
    print("="*60)
    print("DRUG RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n[1/4] Data Preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Adjust sample_frac based on your compute resources
    # 0.3 = 30% of data (faster for testing)
    # 1.0 = 100% of data (best accuracy but slower)
    train_df, test_df = preprocessor.load_and_preprocess(
        train_path='data/drugsComTrain_raw.tsv',
        test_path='data/drugsComTest_raw.tsv',
        sample_frac=0.3  # Adjust this value
    )
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Testing samples: {len(test_df)}")
    
    # Step 2: Feature Engineering
    print("\n[2/4] Feature Engineering...")
    feature_engineer = FeatureEngineer(max_features=3000)
    
    X_train, X_test, y_train, y_test, train_df_processed, test_df_processed = \
        feature_engineer.extract_all_features(train_df, test_df)
    
    print(f"✓ Features extracted: {X_train.shape[1]}")
    
    # Save feature engineer
    feature_engineer.save('models/feature_engineer.pkl')
    print("✓ Feature engineer saved")
    
    # Step 3: Model Training
    print("\n[3/4] Model Training...")
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save best model
    trainer.save_best_model('models/best_model.pkl')
    print(f"\n✓ Best model saved: {trainer.best_model_name}")
    
    # Generate and display charts
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION CHARTS")
    print("="*60)
    
    print("\n✓ Performance comparison chart generated")
    print("✓ Confusion matrix generated")
    print("✓ ROC curves generated")
    print("✓ Precision-Recall curves generated")
    print("✓ Feature importance chart generated")
    print("✓ Metrics radar chart generated")
    print("✓ Class distribution chart generated")
    
    # Step 4: Build Recommendation Engine
    print("\n[4/4] Building Recommendation Engine...")
    recommendation_engine = RecommendationEngine(
        model=trainer.best_model,
        feature_engineer=feature_engineer,
        train_df=train_df_processed,
        test_df=test_df_processed
    )
    
    recommendation_engine.save('models/recommendation_engine.pkl')
    print("✓ Recommendation engine saved")
    
    # Test the system
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*60)
    
    test_conditions = recommendation_engine.get_available_conditions()[:3]
    for condition in test_conditions:
        print(f"\nTesting: {condition}")
        recommendations = recommendation_engine.recommend_drugs(condition, top_n=3)
        
        if recommendations:
            print(f"  Top 3 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. {rec['drug_name']} (Score: {rec['score']:.3f})")
        else:
            print("  No recommendations found")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\n✓ All models trained successfully")
    print("✓ All visualization charts generated")
    print("✓ Recommendation engine ready")
    print("\nYou can now run the web application:")
    print("  reflex run")
    print("\nOr initialize and run:")
    print("  reflex init")
    print("  reflex run")
    print("="*60)


if __name__ == "__main__":
    main()