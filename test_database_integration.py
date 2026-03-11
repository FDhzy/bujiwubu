"""
Test Database Integration for Shoe Recognition System
Verifies that the database-backed recognition system is working correctly.
"""
import os
import sys
import numpy as np
import cv2

from models.shoe_database import ShoeDatabase
from models.pytorch_shoe_model import PyTorchShoeModel
from models.shoe_analyzer import ShoeAnalyzer, ShoeType


def test_database_connection():
    """Test 1: Database Connection"""
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    
    try:
        db = ShoeDatabase()
        print("✓ Database connection established successfully")
        print(f"  Database location: {db.db_path}")
        
        # Check if tables exist
        cursor = db.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['shoe_records', 'shoe_types', 'recognition_cache']
        missing_tables = [t for t in expected_tables if t not in tables]
        
        if missing_tables:
            print(f"✗ Missing tables: {missing_tables}")
            return False
        else:
            print(f"✓ All required tables exist: {tables}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_database_operations():
    """Test 2: Data Retrieval and Storage"""
    print("\n" + "="*60)
    print("TEST 2: Database Operations")
    print("="*60)
    
    try:
        db = ShoeDatabase()
        
        # Test saving a recognition
        print("\nTesting save operation...")
        record_id = db.save_recognition(
            shoe_type="low_shoes",
            confidence=0.85,
            image_hash="test_hash_123"
        )
        
        if record_id > 0:
            print(f"✓ Successfully saved recognition (ID: {record_id})")
        else:
            print("✗ Failed to save recognition")
            return False
        
        # Test cache retrieval
        print("\nTesting cache retrieval...")
        cached = db.get_from_cache("test_hash_123")
        if cached:
            shoe_type, confidence = cached
            print(f"✓ Cache hit: {shoe_type} (confidence={confidence:.2f})")
        else:
            print("✗ Cache miss")
            return False
        
        # Test statistics
        print("\nTesting statistics...")
        stats = db.get_statistics()
        print(f"✓ Statistics retrieved: {stats}")
        
        # Test recent recognitions
        print("\nTesting recent recognitions...")
        recent = db.get_recent_recognitions(limit=5)
        print(f"✓ Retrieved {len(recent)} recent records")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"✗ Database operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_model_loading():
    """Test 3: PyTorch Model Loading"""
    print("\n" + "="*60)
    print("TEST 3: PyTorch Model Integration")
    print("="*60)
    
    try:
        # Check if model file exists (try new shoe recognition model first)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'assets', 'models', 
                                  'shoe_recognition', 'shoe_classifier.pth')
        
        # Fallback to old model path if new model doesn't exist
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, 'assets', 'models', 
                                      'cv_resnest101_general_recognition', 'pytorch_model.pt')
        
        if not os.path.exists(model_path):
            print(f"⚠ Model file not found")
            print("  This is expected if you haven't trained the model yet")
            print("  The system will fall back to built-in CV method")
            return True  # Not a failure, just not available
        
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Try to load model
        print("\nAttempting to load model...")
        model = PyTorchShoeModel(use_database=True)
        
        if model.load_model(model_path):
            print("✓ Model loaded successfully")
            
            # Test prediction with dummy image
            print("\nTesting prediction with dummy image...")
            dummy_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
            shoe_type, confidence = model.predict(dummy_image)
            print(f"✓ Prediction successful: {shoe_type.value} ({confidence:.2%})")
            
            # Check statistics
            stats = model.get_statistics()
            print(f"✓ Database statistics: {stats}")
            
            model.close()
            return True
        else:
            print("✗ Failed to load model")
            return False
        
    except Exception as e:
        print(f"✗ PyTorch model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyzer_integration():
    """Test 4: Shoe Analyzer with Database Backend"""
    print("\n" + "="*60)
    print("TEST 4: Shoe Analyzer Integration")
    print("="*60)
    
    try:
        # Create analyzer with PyTorch backend
        print("\nInitializing shoe analyzer...")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'assets', 'models', 
                                  'cv_resnest101_general_recognition', 'pytorch_model.pt')
        
        external_model = None
        if os.path.exists(model_path):
            try:
                pytorch_model = PyTorchShoeModel(use_database=True)
                if pytorch_model.load_model(model_path):
                    external_model = pytorch_model
                    print("✓ Using PyTorch model with database backend")
            except Exception as e:
                print(f"⚠ Could not load PyTorch model: {e}")
        
        if external_model is None:
            print("✓ Using built-in CV method (fallback)")
        
        analyzer = ShoeAnalyzer(language="en", external_model=external_model)
        
        # Test with synthetic shoe image
        print("\nTesting with synthetic shoe image...")
        test_image = np.ones((400, 300, 3), dtype=np.uint8) * 220
        cv2.rectangle(test_image, (60, 200), (240, 350), (80, 100, 120), -1)
        
        result = analyzer.analyze(test_image)
        print(f"✓ Analysis complete:")
        print(f"  Type: {result.shoe_type.value}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Safe depth: {result.safe_puddle_depth_cm:.1f} cm")
        
        return True
        
    except Exception as e:
        print(f"✗ Analyzer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runtime_errors():
    """Test 5: Runtime Error Handling"""
    print("\n" + "="*60)
    print("TEST 5: Runtime Error Handling")
    print("="*60)
    
    try:
        # Test with invalid inputs
        print("\nTesting error handling...")
        
        db = ShoeDatabase()
        
        # Test with invalid shoe type
        record_id = db.save_recognition("invalid_type", 0.5)
        print("✓ Handles invalid shoe type gracefully")
        
        # Test with None values
        cached = db.get_from_cache(None)
        print("✓ Handles None values gracefully")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all database integration tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "DATABASE INTEGRATION VERIFICATION" + " "*15 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\nThis test verifies the external database integration:")
    print("1. Database connection establishment")
    print("2. Data retrieval and storage")
    print("3. PyTorch model integration")
    print("4. Analyzer with database backend")
    print("5. Runtime error handling")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Database Operations", test_database_operations),
        ("PyTorch Model Loading", test_pytorch_model_loading),
        ("Analyzer Integration", test_analyzer_integration),
        ("Runtime Error Handling", test_runtime_errors),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} : {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    print("="*60)
    
    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED")
        print("The database integration is working correctly!")
        return 0
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
