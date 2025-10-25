@echo off
REM ============================================================================
REM Copy gRPC Runtime DLLs to Sierra Chart
REM ============================================================================

echo Copying gRPC runtime DLLs to Sierra Chart...
echo.

set VCPKG_BIN=C:\vcpkg\installed\x64-windows\bin
set SIERRA=C:\SierraChart

if not exist "%VCPKG_BIN%" (
    echo ERROR: vcpkg bin folder not found
    pause
    exit /b 1
)

if not exist "%SIERRA%" (
    echo ERROR: Sierra Chart folder not found
    pause
    exit /b 1
)

echo Copying DLLs...
echo.

REM Copy all DLLs from vcpkg to Sierra Chart
xcopy /Y "%VCPKG_BIN%\*.dll" "%SIERRA%\"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to copy DLLs
    pause
    exit /b 1
)

echo.
echo SUCCESS! Runtime DLLs copied to Sierra Chart
echo.
echo Next steps:
echo 1. Restart Sierra Chart
echo 2. Analysis -^> Studies
echo 3. Add "NEXUS AI - Complete Integration"
echo.
pause
