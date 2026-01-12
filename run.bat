@echo off
echo Starting Speech-to-Text Service...

REM Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
