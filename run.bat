@echo off
REM run.bat — SkyRoute launcher
REM
REM SAFE TO COMMIT: this script contains no secrets.
REM All API keys live in .env (gitignored). Copy .env.example -> .env and fill in your keys.
REM
REM Usage:
REM   run.bat          start the app
REM   run.bat --check  verify keys are loaded without starting the app

if not exist ".env" (
    echo ERROR: .env not found.
    echo Run:  copy .env.example .env   then fill in your API keys.
    exit /b 1
)

REM Load KEY=VALUE lines from .env (comment lines have no = so they are skipped automatically)
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%B"=="" set "%%A=%%B"
)

REM Report key status
echo.
if defined TOMTOM_API_KEY (
    echo [OK]      TOMTOM_API_KEY loaded
) else (
    echo [MISSING] TOMTOM_API_KEY ^(TomTom traffic routing - register free at developer.tomtom.com^)
)
if defined GROQ_API_KEY (
    echo [OK]      GROQ_API_KEY loaded
) else (
    echo [MISSING] GROQ_API_KEY ^(Route Assistant LLM - register free at console.groq.com^)
)
if defined ANTHROPIC_API_KEY echo [OK]      ANTHROPIC_API_KEY loaded
if defined FAA_API_KEY       echo [OK]      FAA_API_KEY loaded
if defined MAPBOX_TOKEN      echo [OK]      MAPBOX_TOKEN loaded
echo.

if "%1"=="--check" exit /b 0

echo Starting SkyRoute...
streamlit run app.py
