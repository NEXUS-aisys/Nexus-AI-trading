@echo off
REM ========================================
REM Generate C# Protobuf Files for NinjaTrader 8
REM ========================================

echo.
echo ========================================
echo NEXUS AI - Generate C# Protobuf Files
echo ========================================
echo.

cd /d "%~dp0..\proto"

echo [1/3] Checking for protoc...
where protoc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: protoc not found!
    echo.
    echo Please install Protocol Buffers:
    echo 1. Download from: https://github.com/protocolbuffers/protobuf/releases
    echo 2. Extract and add to PATH
    echo.
    pause
    exit /b 1
)

echo [2/3] Generating C# files from nexus_trading.proto...
protoc --csharp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_csharp_plugin.exe nexus_trading.proto

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to generate C# files
    echo.
    echo Make sure grpc_csharp_plugin.exe is in PATH or same directory as protoc
    pause
    exit /b 1
)

echo [3/3] Copying to NinjaTrader directory...

set NT8_DIR=%USERPROFILE%\Documents\NinjaTrader 8\bin\Custom\Indicators

if not exist "%NT8_DIR%" (
    echo WARNING: NinjaTrader 8 directory not found!
    echo Expected: %NT8_DIR%
    echo.
    echo Files generated in proto\ directory
    echo Please copy manually to NinjaTrader
    pause
    exit /b 0
)

copy /Y "NexusTrading.cs" "%NT8_DIR%\" >nul
copy /Y "NexusTradingGrpc.cs" "%NT8_DIR%\" >nul
copy /Y "..\ninjatrader\NexusAI_NT8.cs" "%NT8_DIR%\" >nul

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Files copied to:
echo %NT8_DIR%
echo.
echo Next steps:
echo 1. Open NinjaTrader 8
echo 2. Go to Tools -^> Compile
echo 3. Fix any errors (install gRPC NuGet packages if needed)
echo 4. Add "NEXUS AI" indicator to chart
echo.
pause
