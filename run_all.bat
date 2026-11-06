@echo off
REM Starts the API and the Streamlit app in two windows (Windows).
REM Run once first:  python -m venv .venv  &&  .venv\Scripts\activate  &&  pip install -r requirements.txt
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
start "Box Counter API" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
timeout /t 5 >nul
start "Box Counter UI" cmd /k "python -m streamlit run frontend/app.py"
