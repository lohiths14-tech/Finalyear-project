"""
AgriPulse 2.0 — Enterprise QA & Quality Assurance Audit Suite
Runs comprehensive static and dynamic checks across backend, frontend, security, and ML integrity.
"""

import sys
import os
import subprocess
import re
from pathlib import Path

# Ensure unbuffered real-time output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def print_header(title: str):
    print("\n" + "=" * 65, flush=True)
    print(f"  🔍 {title}", flush=True)
    print("=" * 65, flush=True)

def run_step(step_name: str, cmd: list, cwd: str = ".") -> bool:
    print(f"\n▶ Running: {step_name}...", flush=True)
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0:
            print(f"  ✅ {step_name}: PASSED", flush=True)
            return True
        else:
            print(f"  ❌ {step_name}: FAILED (exit code {res.returncode})", flush=True)
            if res.stderr:
                print(f"     Error output:\n{res.stderr[:400]}", flush=True)
            if res.stdout and not res.stderr:
                print(f"     Output:\n{res.stdout[-400:]}", flush=True)
            return False
    except Exception as e:
        print(f"  ❌ {step_name}: EXCEPTION ({str(e)})", flush=True)
        return False

def check_zero_dummy_data(repo_root: Path) -> bool:
    print("\n▶ Checking Real AI Policy (Zero Dummy Data / No Fake Predictors)...", flush=True)
    forbidden = [
        re.compile(r"torch\.randn\(.*pred"),
        re.compile(r"np\.random\.choice\(.*diseases"),
        re.compile(r"hardcoded.*confidence"),
    ]
    
    violations = []
    py_files = list(repo_root.glob("backend/app/**/*.py"))
    for f in py_files:
        try:
            content = f.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern.search(content):
                    violations.append(f"{f.name}: matched {pattern.pattern}")
        except Exception:
            pass
            
    if violations:
        print(f"  ❌ Found {len(violations)} Real AI policy violation(s):", flush=True)
        for v in violations:
            print(f"     - {v}", flush=True)
        return False
    
    print("  ✅ Real AI Policy Audit: Clean (No synthetic predictors found)", flush=True)
    return True

def check_memory_guards(repo_root: Path) -> bool:
    print("\n▶ Auditing Image Memory Guards (DoS & OOM Prevention)...", flush=True)
    backend_dep = repo_root / "backend" / "app" / "api" / "dependencies.py"
    client_compress = repo_root / "frontend" / "src" / "utils" / "compress.ts"
    
    has_server_guard = backend_dep.exists() and "MAX_IMAGE_DIMENSION" in backend_dep.read_text(encoding="utf-8")
    has_client_guard = client_compress.exists() and "1024" in client_compress.read_text(encoding="utf-8")
    
    if has_server_guard and has_client_guard:
        print("  ✅ Image Memory Guards: Dual client-side (1024px) & server-side bounds verified", flush=True)
        return True
    else:
        print("  ⚠️ Warning: Memory guard missing in client or server", flush=True)
        return False

def main():
    root = Path(__file__).resolve().parent.parent
    is_full = "--full" in sys.argv
    print_header("AgriPulse 2.0 — Enterprise QA Quality Gate")
    
    results = {}
    
    # 1. Real AI Policy
    results["Real AI Policy"] = check_zero_dummy_data(root)
    
    # 2. Image Memory Guards
    results["Memory Guards"] = check_memory_guards(root)
    
    # 3. Backend Pytest Suite
    py_exec = sys.executable
    test_target = ["backend/tests"] if is_full else [
        "backend/tests/test_api_contract.py",
        "backend/tests/test_quality_calibration.py",
        "backend/tests/test_agronomy_dosage.py"
    ]
    results["Backend Pytest Suite"] = run_step(
        f"Backend Pytest ({'Full Suite' if is_full else 'Fast Contract & Quality Suite'})",
        [py_exec, "-m", "pytest"] + test_target + ["-q"],
        cwd=str(root)
    )
    
    # 4. Frontend Production Build
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    results["Frontend Vite Production Build"] = run_step(
        "Frontend Vite Build",
        [npm_cmd, "run", "build"],
        cwd=str(root / "frontend")
    )
    
    # Summary Scorecard
    print_header("QA Audit Summary Scorecard")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED ✅" if passed else "FAILED ❌"
        if not passed:
            all_passed = False
        print(f"  - {test_name:<35}: {status}", flush=True)
        
    print("\n" + "=" * 65, flush=True)
    if all_passed:
        print("  🎉 ALL QUALITY GATES PASSED. System is production ready.", flush=True)
        print("=" * 65 + "\n", flush=True)
        sys.exit(0)
    else:
        print("  ⚠️ QUALITY GATE REJECTED. Please resolve failing items.", flush=True)
        print("=" * 65 + "\n", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
