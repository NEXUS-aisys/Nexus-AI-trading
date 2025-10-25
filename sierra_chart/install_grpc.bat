@echo off
REM ============================================================================
REM Install gRPC and Protobuf via vcpkg
REM This will take 30-60 minutes
REM ============================================================================

echo ============================================
echo Installing gRPC and Protobuf
echo ============================================
echo.
echo This will take 30-60 minutes...
echo Please be patient!
echo.

REM Check if vcpkg exists
if not exist "C:\vcpkg\vcpkg.exe" (
    echo ERROR: vcpkg not found at C:\vcpkg
    echo.
    echo Please install vcpkg first:
    echo   cd C:\
    echo   git clone https://github.com/Microsoft/vcpkg.git
    echo   cd vcpkg
    echo   bootstrap-vcpkg.bat
    pause
    exit /b 1
)

cd C:\vcpkg

echo Installing gRPC for x64-windows...
echo.

vcpkg install grpc:x64-windows

if errorlevel 1 (
    echo.
    echo ERROR: gRPC installation failed
    pause
    exit /b 1
)

echo.
echo Installing Protobuf for x64-windows...
echo.

vcpkg install protobuf:x64-windows

if errorlevel 1 (
    echo.
    echo ERROR: Protobuf installation failed
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo gRPC and Protobuf are now installed.
echo You can now build the Sierra Chart DLL.
echo.
pause
