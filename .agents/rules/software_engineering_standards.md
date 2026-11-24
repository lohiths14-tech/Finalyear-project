# Software Engineering & Quality Assurance Standards

## 1. Zero Broken Builds & Strict Type Safety
- **Frontend**: All code must pass `npm run build` with zero TypeScript or JSX compile errors.
- **Backend**: All FastAPI route handlers, Pydantic models, and pipeline classes must be strictly typed.
- **No Unchecked Exceptions**: Any async call or API request must be enclosed in `try...except` (backend) or `.catch()` / `try...catch` (frontend) with meaningful user-facing error feedback.

## 2. Test-Driven Verification & Regression Blocking
- Every new feature or bugfix must have an associated test in `backend/tests/` or component verification in `frontend/`.
- No pull request or deployable commit is valid without running the full test suite (`pytest backend/tests`).
- Preservation of existing behavior: Never regress previously tested endpoints or UI tabs.

## 3. High-Performance & Memory-Safe Operations
- **Image Upload Bounds**: All image uploads are capped at 1024px maximum dimension on client-side before submission (`frontend/src/utils/compress.ts`) and guarded server-side (`MAX_IMAGE_DIMENSION = 1024`).
- **WebRTC Camera Streams**: Camera streams must explicitly release camera hardware tracks on modal close (`stream.getTracks().forEach(t => t.stop())`).
- **No Memory Leaks**: Component subscriptions and intervals (`setInterval`, event listeners) must have clean unmount teardown in `useEffect`.

## 4. Dual-Theme Visual Perfection
- Every component must render cleanly in both **Dark Mode** (Obsidian #07090e) and **Light Mode** (Alabaster #f8fafc).
- Never use hardcoded light-only backgrounds (e.g. `#ffffff`, `#f0fdf4`, `#fffbeb`) in inline styles that cause blinding white boxes in dark mode. Use CSS custom properties (`var(--bg-card)`, `var(--primary-glow)`, `var(--border)`).

## 5. Security & Boundary Hardening
- Never expose API keys, internal credentials, or private model paths in client bundles or public endpoints.
- Rate-limiting, CORS sanitization, and input file validation (MIME-type check and dimension bounds) are non-negotiable.
- Defense against Out-of-Distribution (OOD): Rejection filters must activate before classification.
