@echo off
echo ========================================
echo   Building Roundtable Executable
echo ========================================
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Building executable...
echo This may take a few minutes...
echo.

pyinstaller roundtable.spec --clean

echo.
echo ========================================
if exist "dist\Roundtable.exe" (
    echo   BUILD SUCCESSFUL!
    echo   Executable: dist\Roundtable.exe
) else (
    echo   BUILD FAILED - Check errors above
)
echo ========================================
echo.
pause
