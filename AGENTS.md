# SmartHire GenAI — Agent Operating Guidelines

These instructions are permanent requirements for any agent working on the SmartHire GenAI repository.

---

## 1. Python Environment & Execution
- **Mandatory Interpreter**: Always use the existing `.venv\Scripts\python.exe` interpreter for executing all Python scripts, tests, package installations, and application execution.
  - Full path: `C:\Users\neshn\OneDrive\Desktop\smarthire-genai\.venv\Scripts\python.exe`
  - Relative command invocation: `.\.venv\Scripts\python.exe`
- **Package Management**: Always use `.\.venv\Scripts\python.exe -m pip` when installing, checking, or managing packages.
- **Prohibited Interpreters**: Never use `venv` or system Python for SmartHire project commands.
- **Environment Lifecycle**: Never create, delete, or recreate virtual environments without explicit user permission.

---

## 2. Dependencies & Package Changes
- **Check Requirements First**: Always inspect `requirements.txt` and verify currently installed packages before installing, updating, or changing packages.
- Do not install or upgrade packages unless strictly necessary and requested.

---

## 3. Code & Configuration Preservation
- **Preserve Existing Logic**: Preserve all existing code, resume parser logic, extracted data, project files, and `.env` configuration.
- Never overwrite or discard existing implementations without explicit authorization.

---

## 4. Development Workflow & Troubleshooting
- **Continuity**: Always continue from the existing project state.
- **Error Transparency**: Always explain any encountered errors clearly before modifying code or proposing fixes.
