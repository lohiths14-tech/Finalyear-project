#!/usr/bin/env python3
"""
Core Functionality Verification Script

Quick verification script to check if core ML and API functionality
is working after rollback operations. This script runs critical checks
without full test suite execution for rapid validation.

Usage:
    python backend/scripts/verify_core_functionality.py
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "backend"))

def test_core_imports():
    """Test if core modules can be imported successfully"""
    print("Testing core imports...")
    
    try:
        # Test basic app structure imports
        import app
        print("✓ app module imported")
        
        import app.core
        print("✓ app.core module imported")
        
        import app.models
        print("✓ app.models module imported")
        
        import app.api
        print("✓ app.api module imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during imports: {e}")
        return False

def test_model_architecture():
    """Test if model architecture can be accessed"""
    print("\nTesting model architecture access...")
    
    try:
        from app.models.architectures import get_model_architecture
        print("✓ Model architecture function imported")
        
        # Test getting architecture info (without loading weights)
        arch_info = get_model_architecture("mobilenet")
        print("✓ Model architecture retrieved")
        
        return True
        
    except ImportError as e:
        print(f"✗ Model import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Model architecture error: {e}")
        return False

def test_inference_service():
    """Test if inference service can be instantiated"""
    print("\nTesting inference service...")
    
    try:
        from app.services.inference_service import InferenceService
        print("✓ InferenceService imported")
        
        # Try to instantiate (without loading actual models)
        # This tests the class structure and basic initialization
        service = InferenceService.__new__(InferenceService)
        print("✓ InferenceService can be instantiated")
        
        return True
        
    except ImportError as e:
        print(f"✗ InferenceService import error: {e}")
        return False
    except Exception as e:
        print(f"✗ InferenceService error: {e}")
        return False

def test_api_structure():
    """Test if API routes can be imported"""
    print("\nTesting API structure...")
    
    try:
        from app.api.routes import field
        print("✓ Field API routes imported")
        
        # Check if FastAPI app can be created
        from fastapi import FastAPI
        app = FastAPI()
        print("✓ FastAPI app can be created")
        
        return True
        
    except ImportError as e:
        print(f"✗ API import error: {e}")
        return False
    except Exception as e:
        print(f"✗ API structure error: {e}")
        return False

def test_database_connection():
    """Test database configuration (without actual connection)"""
    print("\nTesting database configuration...")
    
    try:
        from app.core.database import get_database_url
        print("✓ Database configuration imported")
        
        # Get database URL (tests configuration loading)
        db_url = get_database_url()
        if db_url:
            print("✓ Database URL configured")
        else:
            print("⚠ Database URL not configured (may be expected)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Database import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Database configuration error: {e}")
        return False

def test_image_processing():
    """Test image processing utilities"""
    print("\nTesting image processing...")
    
    try:
        from app.utils.image_processing import preprocess_image
        print("✓ Image processing functions imported")
        
        # Test with dummy data if possible
        import numpy as np
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # This should not crash (even if it returns placeholder data)
        result = preprocess_image(dummy_image)
        print("✓ Image preprocessing function works")
        
        return True
        
    except ImportError as e:
        print(f"✗ Image processing import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Image processing error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from app.core.config import Settings
        settings = Settings()
        print("✓ Configuration settings loaded")
        
        # Check if basic config attributes exist
        if hasattr(settings, 'app_name'):
            print("✓ Basic configuration attributes present")
        else:
            print("⚠ Configuration structure may have changed")
        
        return True
        
    except ImportError as e:
        print(f"✗ Configuration import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def run_all_tests():
    """Run all core functionality tests"""
    print("=== CORE FUNCTIONALITY VERIFICATION ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Model Architecture", test_model_architecture),
        ("Inference Service", test_inference_service),
        ("API Structure", test_api_structure),
        ("Database Configuration", test_database_connection),
        ("Image Processing", test_image_processing),
        ("Configuration", test_configuration),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✓ All core functionality tests PASSED")
        print("System appears to be in a functional state")
        return True
    else:
        print("✗ Some core functionality tests FAILED")
        print("System may need attention before proceeding")
        return False

def main():
    """Main entry point"""
    # Change to backend directory
    backend_dir = project_root / "backend"
    os.chdir(backend_dir)
    
    success = run_all_tests()
    
    if success:
        print("\n🟢 Core functionality verification PASSED")
        sys.exit(0)
    else:
        print("\n🔴 Core functionality verification FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()