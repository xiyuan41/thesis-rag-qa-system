@echo off
setlocal EnableExtensions

rem Run relative commands from the directory containing this launcher.
set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%" || (
    echo [ERROR] Cannot open project directory: "%PROJECT_ROOT%"
    pause
    exit /b 1
)

rem Prefer the project environment; otherwise use system Python if it has the dependencies.
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -c "import uvicorn,streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python, uvicorn, or streamlit is unavailable in the selected environment.
    echo Install the dependencies from requirements.txt, or create the project .venv as described in README.md.
    popd
    pause
    exit /b 1
)

rem Refuse to start duplicate services or interfere with unrelated processes.
for %%P in (8000 8501) do (
    "%PYTHON%" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',int(sys.argv[1]))) == 0 else 1)" %%P >nul 2>&1
    if not errorlevel 1 (
        echo [ERROR] Port %%P is already in use. Close the existing project service and try again.
        popd
        pause
        exit /b 1
    )
)

rem Start backend and frontend in separate persistent windows.
start "Paper RAG Backend" /D "%PROJECT_ROOT%" "%ComSpec%" /k ""%PYTHON%" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
start "Paper RAG Frontend" /D "%PROJECT_ROOT%" "%ComSpec%" /k ""%PYTHON%" -m streamlit run frontend\streamlit_app.py --server.port 8501"

rem Wait up to 60 seconds for the backend before opening the demo page.
echo Waiting for the backend to become ready...
for /L %%I in (1,1,60) do (
    "%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=1)" >nul 2>&1
    if not errorlevel 1 goto backend_ready
    timeout /t 1 /nobreak >nul
)

echo [ERROR] Backend did not become ready within 60 seconds. Check the Paper RAG Backend window.
popd
pause
exit /b 1

:backend_ready
start "" "http://localhost:8501"
echo Project started. Demo page: http://localhost:8501
echo Keep both service windows open during the demonstration.
popd
exit /b 0
