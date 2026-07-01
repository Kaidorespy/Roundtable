@echo off
echo ========================================
echo   Building Roundtable Executable
echo ========================================
echo.

set PYTHON_CMD=python
if exist "build_venv\Scripts\python.exe" (
    set PYTHON_CMD=build_venv\Scripts\python.exe
)

echo Using Python: %PYTHON_CMD%
echo.

REM Make sure the build environment has current dependencies.
%PYTHON_CMD% -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo Failed to install build dependencies.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
%PYTHON_CMD% -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON_CMD% -m pip install pyinstaller
)

echo.
echo Building executable...
echo This may take a few minutes...
echo.

%PYTHON_CMD% -m PyInstaller roundtable.spec --clean --distpath dist_slim --workpath build_slim

echo.
echo ========================================
if exist "dist_slim\Roundtable.exe" (
    echo   BUILD SUCCESSFUL!
    echo   Executable: dist_slim\Roundtable.exe
) else (
    echo   BUILD FAILED - Check errors above
)
echo ========================================
echo.
pause
