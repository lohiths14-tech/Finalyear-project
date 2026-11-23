"""
Property-Based Test for Code Quality Bug Condition Exploration

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**

This test is designed to FAIL on unfixed code to confirm the bug exists.
When it fails, it validates that the code quality violations are present.
When it passes (after fixes), it confirms the expected behavior is achieved.
"""

import subprocess
import pytest
from pathlib import Path
import re
from typing import List, Dict, Tuple
import sys
import os

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


class CodeQualityViolation:
    """Represents a single code quality violation found by linting tools."""
    def __init__(self, file_path: str, line: int, col: int, code: str, message: str):
        self.file_path = file_path
        self.line = line
        self.col = col
        self.code = code
        self.message = message
        
    def __str__(self):
        return f"{self.file_path}:{self.line}:{self.col}: {self.code} {self.message}"


class LintingResults:
    """Container for linting analysis results."""
    def __init__(self, violations: List[CodeQualityViolation]):
        self.violations = violations
        self._categorize_violations()
    
    def _categorize_violations(self):
        """Categorize violations by type for analysis."""
        self.line_length_violations = [v for v in self.violations if v.code == 'E501']
        self.unused_imports = [v for v in self.violations if v.code == 'F401']
        self.formatting_errors = [v for v in self.violations if v.code in ['W291', 'W293', 'E261', 'E302', 'E303', 'E305']]
        self.style_violations = [v for v in self.violations if v.code in ['E127', 'E128', 'E402', 'E701', 'E741']]
        self.unused_variables = [v for v in self.violations if v.code == 'F841']
        self.whitespace_issues = [v for v in self.violations if v.code in ['W291', 'W293']]
        self.redefinitions = [v for v in self.violations if v.code == 'F811']
        
    def has_line_length_violations(self) -> bool:
        """Check if code has line length violations (E501)."""
        return len(self.line_length_violations) > 0
    
    def has_unused_imports(self) -> bool:
        """Check if code has unused imports (F401)."""
        return len(self.unused_imports) > 0
    
    def has_formatting_errors(self) -> bool:
        """Check if code has formatting errors (W291, W293, E261, E302, E303, E305)."""
        return len(self.formatting_errors) > 0
    
    def has_style_violations(self) -> bool:
        """Check if code has style violations (E127, E128, E402, E701, E741)."""
        return len(self.style_violations) > 0
    
    def has_unused_variables(self) -> bool:
        """Check if code has unused variables (F841)."""
        return len(self.unused_variables) > 0
    
    def has_whitespace_issues(self) -> bool:
        """Check if code has whitespace issues (W291, W293)."""
        return len(self.whitespace_issues) > 0
    
    def has_redefinitions(self) -> bool:
        """Check if code has variable redefinitions (F811)."""
        return len(self.redefinitions) > 0
    
    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by category."""
        return {
            'line_length_violations': len(self.line_length_violations),
            'unused_imports': len(self.unused_imports), 
            'formatting_errors': len(self.formatting_errors),
            'style_violations': len(self.style_violations),
            'unused_variables': len(self.unused_variables),
            'whitespace_issues': len(self.whitespace_issues),
            'redefinitions': len(self.redefinitions),
            'total_violations': len(self.violations)
        }


def run_flake8_analysis(target_dir: str = "app") -> LintingResults:
    """Run flake8 linting analysis on the target directory."""
    try:
        # Run flake8 with specific configuration matching expected violations
        cmd = [
            sys.executable, "-m", "flake8", 
            target_dir,
            "--max-line-length=79",  # PEP 8 standard
            "--count",
            "--statistics"
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=backend_dir
        )
        
        # Parse flake8 output to extract violations
        violations = []
        
        for line in result.stdout.split('\n'):
            if line.strip() and ':' in line:
                # Parse format: file:line:col: code message
                match = re.match(r'^([^:]+):(\d+):(\d+):\s+(\w+)\s+(.+)$', line)
                if match:
                    file_path, line_num, col_num, code, message = match.groups()
                    violation = CodeQualityViolation(
                        file_path=file_path,
                        line=int(line_num),
                        col=int(col_num),
                        code=code,
                        message=message
                    )
                    violations.append(violation)
        
        return LintingResults(violations)
        
    except FileNotFoundError:
        # If flake8 not available, try ruff as alternative
        return run_ruff_analysis(target_dir)


def run_ruff_analysis(target_dir: str = "app") -> LintingResults:
    """Run ruff linting analysis as alternative to flake8."""
    try:
        cmd = [
            sys.executable, "-m", "ruff", "check", 
            target_dir,
            "--line-length=79"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True, 
            text=True,
            cwd=backend_dir
        )
        
        violations = []
        for line in result.stdout.split('\n'):
            if line.strip() and ':' in line:
                # Parse ruff format
                match = re.match(r'^([^:]+):(\d+):(\d+):\s+(\w+)\s+(.+)$', line)
                if match:
                    file_path, line_num, col_num, code, message = match.groups()
                    violation = CodeQualityViolation(
                        file_path=file_path,
                        line=int(line_num),
                        col=int(col_num), 
                        code=code,
                        message=message
                    )
                    violations.append(violation)
        
        return LintingResults(violations)
        
    except FileNotFoundError:
        pytest.skip("Neither flake8 nor ruff available for linting analysis")


# Property-Based Tests for Bug Condition Detection

@pytest.mark.property
def test_property_bug_condition_code_quality_violations():
    """
    **Property 1: Bug Condition** - Code Quality Violations Detection
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**
    
    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    
    For any source file where linting violations exist (isBugCondition returns true), 
    the code SHALL contain specific types of violations that prevent it from meeting 
    professional code quality standards.
    
    This property test surfaces counterexamples that demonstrate:
    - Line length violations (E501) exceeding 79 characters  
    - Unused imports (F401) not removed from code
    - Formatting errors (W291, W293, E261, E302, E303, E305)
    - Style violations (E127, E128, E402, E701, E741)
    - Unused variables (F841) assigned but never referenced
    - Variable redefinitions (F811) shadowing imports
    
    Expected Outcome on UNFIXED code: TEST FAILS (this confirms bug exists)
    Expected Outcome on FIXED code: TEST PASSES (confirms bug is resolved)
    """
    
    print("\n=== CODE QUALITY BUG CONDITION EXPLORATION ===")
    print("Running linting analysis to detect code quality violations...")
    
    # Run comprehensive linting analysis
    results = run_flake8_analysis("app")
    
    # Get violation summary for reporting
    summary = results.get_violation_summary()
    
    print(f"\n=== LINTING ANALYSIS RESULTS ===")
    print(f"Total violations found: {summary['total_violations']}")
    print(f"Line length violations (E501): {summary['line_length_violations']}")
    print(f"Unused imports (F401): {summary['unused_imports']}")
    print(f"Formatting errors: {summary['formatting_errors']}")
    print(f"Style violations: {summary['style_violations']}")  
    print(f"Unused variables (F841): {summary['unused_variables']}")
    print(f"Whitespace issues: {summary['whitespace_issues']}")
    print(f"Variable redefinitions (F811): {summary['redefinitions']}")
    
    # Document specific counterexamples found
    if results.violations:
        print(f"\n=== COUNTEREXAMPLES (First 20 violations) ===")
        for i, violation in enumerate(results.violations[:20]):
            print(f"{i+1:2d}. {violation}")
        
        if len(results.violations) > 20:
            print(f"... and {len(results.violations) - 20} more violations")
    
    # The bug condition property: Code quality violations should NOT exist
    # This assertion will FAIL on unfixed code (proving the bug exists)
    # This assertion will PASS on fixed code (proving the bug is resolved)
    
    print(f"\n=== PROPERTY ASSERTION ===")
    print("Testing Property: Code SHALL NOT contain linting violations")
    print("Expected on unfixed code: ASSERTION FAILS (confirms bug exists)")
    print("Expected on fixed code: ASSERTION PASSES (confirms bug resolved)")
    
    # Assert the expected behavior: No linting violations should exist
    assert not results.has_line_length_violations(), (
        f"COUNTEREXAMPLE: Found {len(results.line_length_violations)} line length violations (E501). "
        f"Example: {results.line_length_violations[0] if results.line_length_violations else 'None'}"
    )
    
    assert not results.has_unused_imports(), (
        f"COUNTEREXAMPLE: Found {len(results.unused_imports)} unused imports (F401). "
        f"Example: {results.unused_imports[0] if results.unused_imports else 'None'}"
    )
    
    assert not results.has_formatting_errors(), (
        f"COUNTEREXAMPLE: Found {len(results.formatting_errors)} formatting errors. "
        f"Example: {results.formatting_errors[0] if results.formatting_errors else 'None'}"
    )
    
    assert not results.has_style_violations(), (
        f"COUNTEREXAMPLE: Found {len(results.style_violations)} style violations. "
        f"Example: {results.style_violations[0] if results.style_violations else 'None'}"
    )
    
    assert not results.has_unused_variables(), (
        f"COUNTEREXAMPLE: Found {len(results.unused_variables)} unused variables (F841). "
        f"Example: {results.unused_variables[0] if results.unused_variables else 'None'}"
    )
    
    assert not results.has_whitespace_issues(), (
        f"COUNTEREXAMPLE: Found {len(results.whitespace_issues)} whitespace issues. "
        f"Example: {results.whitespace_issues[0] if results.whitespace_issues else 'None'}"
    )
    
    assert not results.has_redefinitions(), (
        f"COUNTEREXAMPLE: Found {len(results.redefinitions)} variable redefinitions (F811). "
        f"Example: {results.redefinitions[0] if results.redefinitions else 'None'}"
    )
    
    # If we reach here, all assertions passed - no code quality violations found
    print("✅ SUCCESS: No code quality violations detected - code meets standards!")


@pytest.mark.property  
def test_property_specific_violation_detection():
    """
    **Property 1b: Specific Bug Condition Cases** - Targeted Violation Detection
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**
    
    This property test validates specific known violation patterns that should
    be present in unfixed code based on the bug analysis.
    
    Expected violations based on design analysis:
    - field.py: unused 'json' import, variable redefinition, unused variables
    - Multiple files: E501 line length violations
    - Multiple files: W293 whitespace in blank lines
    - leaf.py: E402 import not at top
    """
    
    print("\n=== SPECIFIC VIOLATION PATTERN DETECTION ===")
    
    # Run analysis
    results = run_flake8_analysis("app")
    
    # Check for specific violations mentioned in the bug analysis
    field_py_violations = [v for v in results.violations if 'field.py' in v.file_path]
    leaf_py_violations = [v for v in results.violations if 'leaf.py' in v.file_path]
    
    print(f"field.py violations: {len(field_py_violations)}")
    print(f"leaf.py violations: {len(leaf_py_violations)}")
    
    # Test specific expected violations (these should exist on unfixed code)
    has_field_unused_import = any(v.code == 'F401' and 'json' in v.message.lower() 
                                  for v in field_py_violations)
    has_field_redefinition = any(v.code == 'F811' and 'json' in v.message.lower() 
                                 for v in field_py_violations)  
    has_field_unused_vars = any(v.code == 'F841' and 'regions' in v.message.lower()
                                for v in field_py_violations)
    has_leaf_import_order = any(v.code == 'E402' for v in leaf_py_violations)
    
    print(f"field.py unused import (F401): {has_field_unused_import}")
    print(f"field.py redefinition (F811): {has_field_redefinition}")  
    print(f"field.py unused variables (F841): {has_field_unused_vars}")
    print(f"leaf.py import order (E402): {has_leaf_import_order}")
    
    # These specific patterns should NOT exist in properly fixed code
    # On unfixed code, these assertions will FAIL (confirming specific bugs exist)
    assert not has_field_unused_import, "COUNTEREXAMPLE: field.py has unused json import (F401)"
    assert not has_field_redefinition, "COUNTEREXAMPLE: field.py has json redefinition (F811)"  
    assert not has_field_unused_vars, "COUNTEREXAMPLE: field.py has unused regions variable (F841)"
    assert not has_leaf_import_order, "COUNTEREXAMPLE: leaf.py has imports not at top (E402)"
    
    print("✅ SUCCESS: No specific violation patterns detected!")


if __name__ == "__main__":
    # Allow running the test directly for debugging
    test_property_bug_condition_code_quality_violations()
    test_property_specific_violation_detection()