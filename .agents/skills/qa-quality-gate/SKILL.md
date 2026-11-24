---
name: qa-quality-gate
description: >-
  Automated software quality assurance and regression blocker for AgriPulse 2.0.
  Executes backend pytest suites, verifies frontend TypeScript builds, validates API contract
  schemas, and checks system health before merging, committing, or deploying.
---

# QA Quality Gate & Regression Blocker

Use this skill whenever you are preparing a commit, implementing a new feature, or validating the integrity of the AgriPulse 2.0 platform.

## Quality Gate Checklist

Every change must satisfy the 5-point QA gate:

1. **Backend Test Suite Passing**:
   ```powershell
   pytest backend/tests -v
   ```
   All unit and property preservation tests must pass without regressions.

2. **Frontend Production Build**:
   ```powershell
   cd frontend; npm run build
   ```
   Must produce clean bundles with zero TypeScript or syntax errors.

3. **API Contract Verification**:
   Verify that all 16 endpoints return valid OpenAPI response schemas without 500 crashes.

4. **Memory & Upload Limits**:
   Verify that image processing functions adhere to `MAX_IMAGE_DIMENSION = 1024` to prevent OOM errors on constrained server environments (Render Free Tier 512MB RAM).

5. **Zero Dummy Data Compliance**:
   Confirm that all predictions, heatmaps, and severity scores originate from authentic inference models per `AGENTS.md`.

## Automated Audit Command
Run the unified QA audit runner:
```powershell
python scripts/qa_audit.py
```
If any check fails, do NOT commit or deploy until the issue is identified and resolved.
