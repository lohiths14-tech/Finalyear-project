#!/usr/bin/env python3
"""
Run Preservation Property Tests for Code Quality Bugfix

This script runs the preservation property tests to establish baseline behavior
on UNFIXED code before implementing code quality improvements.

EXPECTED OUTCOME: All tests should PASS (confirms baseline to preserve)
"""

import subprocess
import sys
import time

def run_test_class(class_name, timeout=60):
    """Run a specific test class with timeout"""
    print(f"\n🧪 Testing {class_name}")
    print("="*50)
    
    cmd = [
        "python", "-m", "pytest", 
        f"test_preservation_properties.py::{class_name}",
        "-v", "--tb=short", "-x"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {class_name}: PASSED")
            return True
        else:
            print(f"❌ {class_name}: FAILED")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {class_name}: TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"💥 {class_name}: ERROR - {e}")
        return False

def main():
    print("🔬 Running Preservation Property Tests on UNFIXED Code")
    print("=" * 60)
    print("Purpose: Establish baseline behavior before code quality fixes")
    print("Expected: All tests PASS (confirms functionality to preserve)")
    print("=" * 60)
    
    # Test classes to run (ordered by complexity/speed)
    test_classes = [
        ("TestMLModelPreservation", 30),          # Just decision engine, not full model
        ("TestGeospatialPreservation", 60),       # GPS coordinate tests
        ("TestDatabaseOperationPreservation", 45), # Database model tests  
        ("TestTrainingPipelinePreservation", 45), # Training config tests
    ]
    
    passed = 0
    total = len(test_classes)
    
    start_time = time.time()
    
    for class_name, timeout in test_classes:
        if run_test_class(class_name, timeout):
            passed += 1
        
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print(f"📊 PRESERVATION TEST RESULTS")
    print(f"Passed: {passed}/{total}")
    print(f"Duration: {duration:.1f}s")
    
    if passed == total:
        print("✅ ALL PRESERVATION TESTS PASSED")
        print("✨ Baseline behavior successfully captured!")
        print("🎯 Ready to proceed with code quality fixes")
        return True
    else:
        print("❌ SOME PRESERVATION TESTS FAILED")
        print("⚠️  Investigation needed before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)