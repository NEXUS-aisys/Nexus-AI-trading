@echo off
REM Generate protobuf C++ files from proto definition

echo Generating protobuf files...
echo.

set VCPKG=C:\vcpkg\installed\x64-windows

REM Check if proto file exists
if not exist "..\proto\nexus_trading.proto" (
    echo ERROR: Proto file not found at ..\proto\nexus_trading.proto
    pause
    exit /b 1
)

REM Generate C++ code
"%VCPKG%\tools\protobuf\protoc.exe" ^
    --cpp_out=. ^
    --grpc_out=. ^
    --plugin=protoc-gen-grpc="%VCPKG%\tools\grpc\grpc_cpp_plugin.exe" ^
    -I..\proto ^
    ..\proto\nexus_trading.proto

if errorlevel 1 (
    echo.
    echo ERROR: Protobuf generation failed
    pause
    exit /b 1
)

echo.
echo SUCCESS! Generated files:
echo   - nexus_trading.pb.h
echo   - nexus_trading.pb.cc
echo   - nexus_trading.grpc.pb.h
echo   - nexus_trading.grpc.pb.cc
echo.
pause
