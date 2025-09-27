@echo off
REM BitNet b1.58 2B4T Installation Script for Windows

echo 🚀 Setting up BitNet b1.58 2B4T testing environment
echo ==================================================

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ uv is not installed. Please install uv first:
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

echo ✅ uv is installed

REM Create virtual environment
echo 📦 Creating virtual environment...
uv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
uv pip install -r requirements.txt

REM Install specific transformers version for BitNet
echo 🔧 Installing BitNet-compatible transformers...
uv pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4

echo ✅ Installation complete!
echo.
echo To activate the environment:
echo   .venv\Scripts\activate.bat
echo.
echo To run tests:
echo   python simple_test.py
echo   python test_bitnet.py
pause
