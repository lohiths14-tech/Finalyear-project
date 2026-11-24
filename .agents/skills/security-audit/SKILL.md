---
name: security-audit
description: >-
  Static security analysis, input boundary inspection, secrets detection, and OWASP API audit
  for FastAPI endpoints and React client state in AgriPulse 2.0.
---

# Security & Surface Audit

Use this skill to inspect and harden code surfaces against vulnerabilities, injection risks, resource exhaustion, and credential leakage.

## Core Audit Targets

### 1. Secret & Key Leak Prevention
- Verify that `.env` or credentials are never checked into Git.
- Ensure API keys (`X-API-Key`, `VITE_API_URL`, Gemini keys) are loaded through secure environment loaders.
- Run pre-commit pattern scan:
  ```powershell
  git status
  ```
  Ensure `.env` or temporary key files are in `.gitignore`.

### 2. Input File Sanitization & DoS Prevention
- Verify image upload MIME types (`image/jpeg`, `image/png`, `image/webp`).
- Verify dimension downscaling guards:
  - Client-side: Canvas compression downscaling to max 1024px before HTTP upload.
  - Server-side: `MAX_IMAGE_DIMENSION` guard preventing decompression bombs.

### 3. Cross-Origin (CORS) & Route Protection
- Check `backend/app/main.py` CORS configuration to ensure legitimate origins are accepted.
- Verify that error responses never expose internal stack traces or directory paths to the client.

### 4. Client State & Hardware Track Teardown
- Ensure WebRTC camera streams (`navigator.mediaDevices.getUserMedia`) are stopped cleanly when modals close.
- Ensure all `useEffect` hooks with timers or network listeners return proper cleanup functions.
