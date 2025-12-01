"""
Verification script to ensure all components are properly set up
Run this before training models or starting the application
"""
import os
import sys

def check_directory_structure():
    """Verify all required directories exist"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    required_dirs = ['data', 'models', 'database', 'drug_recommendation_system']
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/ exists")
        else:
            print(f"✗ {directory}/ MISSING")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        print("Creating missing directories...")
        for directory in missing_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created {directory}/")
    else:
        print("\n✅ All directories exist")
    
    return len(missing_dirs) == 0


def check_dataset_files():
    """Verify dataset files are present"""
    print("\n" + "="*60)
    print("CHECKING DATASET FILES")
    print("="*60)
    
    train_file = 'data/drugsComTrain_raw.tsv'
    test_file = 'data/drugsComTest_raw.tsv'
    
    if os.path.exists(train_file):
        size = os.path.getsize(train_file) / (1024 * 1024)  # MB
        print(f"✓ {train_file} exists ({size:.1f} MB)")
    else:
        print(f"✗ {train_file} MISSING")
        print("  Download from: https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com")
        return False
    
    if os.path.exists(test_file):
        size = os.path.getsize(test_file) / (1024 * 1024)  # MB
        print(f"✓ {test_file} exists ({size:.1f} MB)")
    else:
        print(f"✗ {test_file} MISSING")
        print("  Download from: https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com")
        return False
    
    print("\n✅ All dataset files present")
    return True


def check_python_files():
    """Verify all required Python files exist"""
    print("\n" + "="*60)
    print("CHECKING PYTHON FILES")
    print("="*60)
    
    required_files = {
        'rxconfig.py': 'Reflex configuration',
        'train_models.py': 'Model training script',
        'database/__init__.py': 'Database package init',
        'database/db_manager.py': 'Database manager',
        'models/__init__.py': 'Models package init',
        'models/data_preprocessing.py': 'Data preprocessing',
        'models/feature_engineering.py': 'Feature engineering',
        'models/model_training.py': 'Model training',
        'models/recommendation_engine.py': 'Recommendation engine',
        'drug_recommendation_system/__init__.py': 'App package init',
        'drug_recommendation_system/drug_recommendation_system.py': 'Main Reflex app'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✓ {file_path} ({description})")
        else:
            print(f"✗ {file_path} MISSING ({description})")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files")
        print("Please create the missing files using the provided artifacts")
        return False
    else:
        print("\n✅ All Python files present")
        return True


def check_dependencies():
    """Verify all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = {
        'reflex': 'Reflex framework',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'lightgbm': 'LightGBM classifier',
        'nltk': 'Natural language toolkit',
        'bs4': 'BeautifulSoup (web scraping)',
        'vaderSentiment': 'VADER sentiment analysis',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} ({description})")
        except ImportError:
            print(f"✗ {package} MISSING ({description})")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} packages")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed")
        return True


def check_nltk_data():
    """Verify NLTK data is downloaded"""
    print("\n" + "="*60)
    print("CHECKING NLTK DATA")
    print("="*60)
    
    try:
        import nltk
        
        required_data = ['stopwords', 'wordnet', 'punkt']
        missing_data = []
        
        for data_name in required_data:
            try:
                nltk.data.find(f'corpora/{data_name}' if data_name != 'punkt' else f'tokenizers/{data_name}')
                print(f"✓ {data_name}")
            except LookupError:
                print(f"✗ {data_name} MISSING")
                missing_data.append(data_name)
        
        if missing_data:
            print(f"\n⚠️  Missing NLTK data: {', '.join(missing_data)}")
            print("Downloading missing data...")
            for data_name in missing_data:
                nltk.download(data_name, quiet=True)
                print(f"✓ Downloaded {data_name}")
            print("\n✅ All NLTK data now available")
        else:
            print("\n✅ All NLTK data present")
        
        return True
    except Exception as e:
        print(f"\n❌ Error checking NLTK data: {e}")
        return False


def check_trained_models():
    """Check if models have been trained"""
    print("\n" + "="*60)
    print("CHECKING TRAINED MODELS")
    print("="*60)
    
    model_files = {
        'models/best_model.pkl': 'Best trained model',
        'models/feature_engineer.pkl': 'Feature engineering pipeline',
        'models/recommendation_engine.pkl': 'Recommendation engine'
    }
    
    all_exist = True
    for file_path, description in model_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file_path} ({description}) - {size:.1f} MB")
        else:
            print(f"✗ {file_path} NOT FOUND ({description})")
            all_exist = False
    
    if all_exist:
        print("\n✅ All models trained and ready")
        return True
    else:
        print("\n⚠️  Models not trained yet")
        print("Run: python train_models.py")
        return False


def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("DRUG RECOMMENDATION SYSTEM - SETUP VERIFICATION")
    print("="*60)
    print("This script will verify your setup is complete and correct")
    
    # Run all checks
    results = {
        'Directory Structure': check_directory_structure(),
        'Dataset Files': check_dataset_files(),
        'Python Files': check_python_files(),
        'Dependencies': check_dependencies(),
        'NLTK Data': check_nltk_data(),
        'Trained Models': check_trained_models()
    }
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYour system is ready. Next steps:")
        print("1. If models not trained: python train_models.py")
        print("2. Initialize Reflex: reflex init")
        print("3. Run application: reflex run")
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        print("Refer to README.md for detailed setup instructions.")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    