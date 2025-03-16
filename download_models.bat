@echo off
setlocal enabledelayedexpansion

REM Script to download ONNX models from HuggingFace for Windows
REM Requires curl.exe (included in Windows 10 1803 and later)

REM Configuration
set ASSETS_DIR=app\src\main\assets

REM Create assets directory if it doesn't exist
if not exist "%ASSETS_DIR%" (
  echo Creating assets directory at %ASSETS_DIR%
  mkdir "%ASSETS_DIR%"
)

echo === Starting ONNX model downloads to %ASSETS_DIR% ===

REM Download function
:download_file
set source_url=%~1
set target_dir=%~2
set filename=%~3
set display_name=%~4

echo Downloading %display_name% to %target_dir%

REM Create target directory if it doesn't exist
if not exist "%target_dir%" mkdir "%target_dir%"

REM Attempt download with retries
set max_attempts=3
set attempt=1

:retry
echo Attempt %attempt% of %max_attempts%...
curl -L "%source_url%" --output "%target_dir%\%filename%" --ssl-no-revoke

if %ERRORLEVEL% EQU 0 (
  echo ✓ Successfully downloaded %filename%
  exit /b 0
) else (
  echo Download attempt %attempt% failed.
  set /a attempt+=1
  if %attempt% LEQ %max_attempts% (
    echo Retrying...
    timeout /t 2 >nul
    goto retry
  ) else (
    echo ❌ Failed to download %filename% after %max_attempts% attempts
    exit /b 1
  )
)
goto :eof

REM Process all models
set total_downloads=6
set current=0
set failed=0

REM Model 1: all-MiniLM-L6-v2
set model_dir=%ASSETS_DIR%\all-minilm-l6-v2
echo.
echo 📦 Processing model: sentence-transformers/all-MiniLM-L6-v2

set /a current+=1
echo [!current!/%total_downloads%] Downloading model.onnx...
call :download_file "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" "%model_dir%" "model.onnx" "all-MiniLM-L6-v2 model"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

set /a current+=1
echo [!current!/%total_downloads%] Downloading tokenizer.json...
call :download_file "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" "%model_dir%" "tokenizer.json" "all-MiniLM-L6-v2 tokenizer"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

REM Model 2: bge-small-en-v1.5
set model_dir=%ASSETS_DIR%\bge-small-en-v1.5
echo.
echo 📦 Processing model: BAAI/bge-small-en-v1.5

set /a current+=1
echo [!current!/%total_downloads%] Downloading model.onnx...
call :download_file "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" "%model_dir%" "model.onnx" "bge-small-en-v1.5 model"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

set /a current+=1
echo [!current!/%total_downloads%] Downloading tokenizer.json...
call :download_file "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json" "%model_dir%" "tokenizer.json" "bge-small-en-v1.5 tokenizer"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

REM Model 3: snowflake-arctic-embed-s
set model_dir=%ASSETS_DIR%\snowflake-arctic-embed-s
echo.
echo 📦 Processing model: Snowflake/snowflake-arctic-embed-s

set /a current+=1
echo [!current!/%total_downloads%] Downloading model.onnx...
call :download_file "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model.onnx" "%model_dir%" "model.onnx" "snowflake-arctic-embed-s model"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

set /a current+=1
echo [!current!/%total_downloads%] Downloading tokenizer.json...
call :download_file "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/tokenizer.json" "%model_dir%" "tokenizer.json" "snowflake-arctic-embed-s tokenizer"
if %ERRORLEVEL% NEQ 0 set /a failed+=1

REM Display summary
echo.
echo === Download Summary ===
echo Total files: %total_downloads%
echo Successfully downloaded: %current%
set /a success=total_downloads-failed

if %failed% GTR 0 (
  echo Failed downloads: %failed%
  exit /b 1
) else (
  echo ✅ All downloads completed successfully
)

endlocal