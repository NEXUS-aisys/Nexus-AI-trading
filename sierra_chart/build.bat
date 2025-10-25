@echo off
REM ============================================================================
REM NEXUS AI - Simple Build Script
REM One-file solution for Sierra Chart
REM ============================================================================

echo ============================================
echo NEXUS AI - Building Sierra Chart DLL
echo ============================================
echo.

REM Check vcpkg
if not exist "C:\vcpkg\installed\x64-windows\include\grpcpp\grpcpp.h" (
    echo ERROR: gRPC not installed via vcpkg
    echo.
    echo Please run:
    echo   cd C:\vcpkg
    echo   vcpkg install grpc:x64-windows protobuf:x64-windows
    pause
    exit /b 1
)

REM Set paths
set VCPKG=C:\vcpkg\installed\x64-windows
set SIERRA=C:\SierraChart

REM Find Visual Studio (supports 2022, 2025, and Insiders)
set VS_FOUND=0

if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
)

if %VS_FOUND%==0 (
    echo ERROR: Visual Studio not found
    echo Checked: VS 2025 Insiders, VS 2022 Community/Professional/Enterprise
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [1/3] Compiling NexusAI_Complete.cpp...
echo.

cl /c /EHsc /std:c++17 /MD /O2 ^
   /I"%VCPKG%\include" ^
   /I"%SIERRA%\ACS_Source" ^
   /D_CRT_SECURE_NO_WARNINGS ^
   /DWIN32_LEAN_AND_MEAN ^
   /DNOMINMAX ^
   /D_WIN32_WINNT=0x0A00 ^
   NexusAI_Complete.cpp

if errorlevel 1 (
    echo.
    echo BUILD FAILED - Compilation error
    pause
    exit /b 1
)

echo [2/3] Compiling protobuf files...
echo.

cl /c /EHsc /std:c++17 /MD /O2 ^
   /I"%VCPKG%\include" ^
   nexus_trading.pb.cc nexus_trading.grpc.pb.cc

if errorlevel 1 (
    echo.
    echo BUILD FAILED - Protobuf compilation error
    pause
    exit /b 1
)

echo [3/3] Linking DLL...
echo.

REM Link with all required libraries
link /DLL /OUT:NexusAI.dll ^
     NexusAI_Complete.obj ^
     nexus_trading.pb.obj ^
     nexus_trading.grpc.pb.obj ^
     /LIBPATH:"%VCPKG%\lib" ^
     grpc++.lib grpc.lib gpr.lib address_sorting.lib re2.lib ^
     upb_base_lib.lib upb_json_lib.lib upb_mem_lib.lib upb_message_lib.lib upb_mini_descriptor_lib.lib upb_textformat_lib.lib upb_wire_lib.lib ^
     libprotobuf.lib cares.lib zlib.lib libssl.lib libcrypto.lib ^
     absl_*.lib ^
     ws2_32.lib

if errorlevel 1 (
    echo.
    echo BUILD FAILED - Linking error
    pause
    exit /b 1
)

echo.
echo ============================================
echo BUILD SUCCESSFUL!
echo ============================================
echo.
echo DLL: NexusAI.dll
echo.

REM Copy to Sierra Chart
if exist "%SIERRA%\Data" (
    copy /Y NexusAI.dll "%SIERRA%\Data\"
    echo Installed to Sierra Chart!
) else (
    echo Copy NexusAI.dll to your Sierra Chart Data folder
)

echo.
pause
