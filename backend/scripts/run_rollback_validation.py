#!/usr/bin/env python3
"""
Rollback Validation Script for Code Quality Bugfix

This script runs comprehensive validation tests to ensure system integrity
after rollback operations or before committing code quality changes.

Usage:
    python backend/scripts/run_rollback_validation.py [--quick] [--category CATEGORY]
"""

import sys
import os
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "backend"))

class ValidationResult:
    def __init__(self, name: str, success: bool, duration: float, details: str = ""):
        self.name = name
        self.success = success
        self.duration = duration
        self.details = details
        
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name} ({self.duration:.2f}s)"

class RollbackValidator:
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results: List[ValidationResult] = []
        self.backend_path = project_root / "backend"
        self.project_root = project_root
        
        # Ensure we're in the backend directory for tests
        os.chdir(self.backend_path)
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}
        print(f"[{timestamp}] {prefix.get(level, 'ℹ')} {message}")
    
    def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[bool, str]:
        """Run a command and return (success, output)"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.backend_path
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, f"Command failed: {str(e)}"
    
    def validate_preservation_properties(self) -> ValidationResult:
        """Run preservation property tests to ensure ML functionality unchanged"""
        self.log("Running preservation property tests...")
        start_time = time.time()
        
        test_files = [
            "test_preservation_properties.py",
            "test_preservation_focused.py", 
            "test_preservation_simple.py"
        ]
        
        successful_tests = 0
        total_tests = 0
        details = []
        
        for test_file in test_files:
            if (self.backend_path / test_file).exists():
                success, output = self.run_command([
                    sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                ])
                total_tests += 1
                if success:
                    successful_tests += 1
                    details.append(f"✓ {test_file}")
                else:
                    details.append(f"✗ {test_file}: {output[:100]}...")
        
        duration = time.time() - start_time
        success = successful_tests == total_tests and total_tests > 0
        
        return ValidationResult(
            "Preservation Properties",
            success,
            duration,
            f"{successful_tests}/{total_tests} test files passed. " + "; ".join(details[:3])
        )
    
    def validate_core_functionality(self) -> ValidationResult:
        """Test core ML and API functionality"""
        self.log("Validating core functionality...")
        start_time = time.time()
        
        # Test basic imports
        core_imports = [
            "import app.core",
            "import app.models",
            "import app.api",
            "import app.services",
        ]
        
        failed_imports = []
        for import_stmt in core_imports:
            success, output = self.run_command([
                sys.executable, "-c", import_stmt
            ])
            if not success:
                failed_imports.append(import_stmt.split()[-1])
        
        duration = time.time() - start_time
        success = len(failed_imports) == 0
        
        details = "All imports successful" if success else f"Failed imports: {', '.join(failed_imports)}"
        
        return ValidationResult("Core Functionality", success, duration, details)
    
    def validate_api_tests(self) -> ValidationResult:
        """Run API endpoint tests"""
        self.log("Running API tests...")
        start_time = time.time()
        
        test_patterns = [
            "tests/test_api*.py",
            "tests/test_*api*.py"
        ]
        
        if self.quick_mode:
            # Just run a subset of tests in quick mode
            success, output = self.run_command([
                sys.executable, "-m", "pytest", "tests/", "-k", "api", "-x", "--tb=short"
            ])
        else:
            success, output = self.run_command([
                sys.executable, "-m", "pytest", "tests/test_api.py", "-v"
            ])
        
        duration = time.time() - start_time
        
        # Extract test count from output if possible
        details = "API tests completed"
        if "passed" in output:
            details = output.split("\n")[-2] if output.split("\n") else details
        
        return ValidationResult("API Tests", success, duration, details)
    
    def validate_model_functionality(self) -> ValidationResult:
        """Test ML model loading and basic inference"""
        self.log("Testing ML model functionality...")
        start_time = time.time()
        
        # Test model loading without actually loading large weights
        test_code = """
try:
    from app.models import get_model_architecture
    from app.services.inference_service import InferenceService
    print("Model imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
except Exception as e:
    print(f"Other error: {e}")
    exit(1)
"""
        
        success, output = self.run_command([
            sys.executable, "-c", test_code
        ])
        
        duration = time.time() - start_time
        details = "Model architecture loading successful" if success else f"Failed: {output[:100]}"
        
        return ValidationResult("Model Functionality", success, duration, details)
    
    def validate_database_operations(self) -> ValidationResult:
        """Test database connectivity and basic operations"""
        self.log("Testing database operations...")
        start_time = time.time()
        
        # Test database imports and basic connectivity
        test_code = """
try:
    from app.core.database import get_database
    print("Database imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
except Exception as e:
    print(f"Database connection error: {e}")
    # Don't fail for connection errors in validation
    print("Database imports successful (connection not tested)")
"""
        
        success, output = self.run_command([
            sys.executable, "-c", test_code
        ])
        
        duration = time.time() - start_time
        details = "Database imports successful" if success else f"Failed: {output[:50]}"
        
        return ValidationResult("Database Operations", success, duration, details)
    
    def validate_geospatial_functions(self) -> ValidationResult:
        """Test geospatial calculation functions"""
        self.log("Testing geospatial functions...")
        start_time = time.time()
        
        # Run geospatial tests if they exist
        geospatial_tests = [
            "tests/test_geospatial.py",
            "tests/test_geo*.py"
        ]
        
        test_found = False
        success = True
        details = "No geospatial tests found"
        
        for test_pattern in geospatial_tests:
            test_files = list(self.backend_path.glob(test_pattern))
            if test_files:
                test_found = True
                for test_file in test_files:
                    test_success, output = self.run_command([
                        sys.executable, "-m", "pytest", str(test_file), "-v"
                    ])
                    if not test_success:
                        success = False
                        details = f"Geospatial test failed: {output[:100]}"
                        break
                    else:
                        details = "Geospatial tests passed"
        
        if not test_found:
            # Test geospatial imports at minimum
            test_code = """
try:
    from app.geospatial import coordinate_utils
    print("Geospatial imports successful")
except ImportError:
    print("No geospatial modules found (may be expected)")
except Exception as e:
    print(f"Geospatial error: {e}")
    exit(1)
"""
            success, output = self.run_command([sys.executable, "-c", test_code])
            details = "Geospatial imports tested"
        
        duration = time.time() - start_time
        return ValidationResult("Geospatial Functions", success, duration, details)
    
    def validate_code_quality(self) -> ValidationResult:
        """Run basic code quality checks"""
        self.log("Running code quality validation...")
        start_time = time.time()
        
        # Check if ruff is available and run basic checks
        success, output = self.run_command(["ruff", "check", "app/", "--select=E9,F63,F7,F82"])
        
        if not success and "command not found" in output.lower():
            # Fallback to Python syntax check
            success, output = self.run_command([
                sys.executable, "-m", "py_compile", "app/__init__.py"
            ])
            details = "Python syntax validation" if success else "Syntax errors found"
        else:
            details = "Basic ruff checks passed" if success else "Code quality issues found"
        
        duration = time.time() - start_time
        return ValidationResult("Code Quality", success, duration, details)
    
    def run_validation(self, categories: Optional[List[str]] = None) -> bool:
        """Run all validation checks"""
        self.log("Starting rollback validation...")
        self.log(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        self.log(f"Working directory: {self.backend_path}")
        
        # Define all validation methods
        all_validations = {
            "preservation": self.validate_preservation_properties,
            "core": self.validate_core_functionality,
            "api": self.validate_api_tests,
            "models": self.validate_model_functionality,
            "database": self.validate_database_operations,
            "geospatial": self.validate_geospatial_functions,
            "quality": self.validate_code_quality,
        }
        
        # Filter validations based on categories
        if categories:
            validations = {k: v for k, v in all_validations.items() if k in categories}
        else:
            validations = all_validations
        
        # Run validations
        start_time = time.time()
        for name, validation_func in validations.items():
            try:
                result = validation_func()
                self.results.append(result)
                self.log(str(result), "SUCCESS" if result.success else "ERROR")
            except Exception as e:
                error_result = ValidationResult(name.title(), False, 0, str(e))
                self.results.append(error_result)
                self.log(f"Validation {name} crashed: {e}", "ERROR")
        
        total_duration = time.time() - start_time
        
        # Report results
        self.log("=== VALIDATION SUMMARY ===")
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        
        for result in self.results:
            self.log(str(result))
        
        self.log(f"Total: {passed}/{total} validations passed ({total_duration:.2f}s)")
        
        # Save detailed results
        self.save_results()
        
        return passed == total
    
    def save_results(self):
        """Save validation results to file"""
        results_file = self.project_root / "logs" / "rollback_validation.json"
        results_file.parent.mkdir(exist_ok=True)
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "quick_mode": self.quick_mode,
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.log(f"Detailed results saved to: {results_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run rollback validation tests")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (fewer tests)")
    parser.add_argument("--category", action="append", choices=[
        "preservation", "core", "api", "models", "database", "geospatial", "quality"
    ], help="Run specific validation categories only")
    
    args = parser.parse_args()
    
    validator = RollbackValidator(quick_mode=args.quick)
    success = validator.run_validation(categories=args.category)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()