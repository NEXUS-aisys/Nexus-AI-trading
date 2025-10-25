@echo off
REM ============================================================================
REM NEXUS AI - CMake Build Script
REM ============================================================================

echo ========================================
echo NEXUS AI - Building with CMake
echo ========================================
echo.

cd /d "%~dp0"

REM Create build directory
if not exist "build" mkdir build
cd build

echo [1/3] Configuring with CMake...
echo.

cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo.
    echo ERROR: CMake configuration failed
    pause
    exit /b 1
)

echo.
echo [2/3] Building...
echo.

cmake --build . --config Release

if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo [3/3] Installing to Sierra Chart...
echo.

cmake --install . --config Release

if errorlevel 1 (
    echo WARNING: Install failed - copy manually
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo DLL Location: build\Release\NexusAI.dll
echo.

if exist "Release\NexusAI.dll" (
    echo Copying to Sierra Chart...
    copy /Y "Release\NexusAI.dll" "C:\SierraChart\Data\" 2>nul
    if errorlevel 1 (
        echo Please manually copy to C:\SierraChart\Data\
    ) else (
        echo Installed successfully!
    )
)

echo.
pause
