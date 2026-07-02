@echo off
REM build.bat — Compile both programs on Windows with MS-MPI
REM Run this on the Master Node (Member 1's laptop)

echo ============================================
echo  Building Sequential Analytics
echo ============================================
cl /EHsc /O2 /std:c++17 sequential_analytics.cpp /out:sequential_analytics.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Sequential build failed.
    pause
    exit /b 1
)
echo [OK] sequential_analytics.exe built.

echo.
echo ============================================
echo  Building MPI Analytics
echo ============================================
cl /EHsc /O2 /std:c++17 mpi_analytics.cpp ^
    /I "%MSMPI_INC%" ^
    /link "%MSMPI_LIB64%\msmpi.lib" ^
    /out:mpi_analytics.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MPI build failed. Make sure MS-MPI SDK is installed.
    pause
    exit /b 1
)
echo [OK] mpi_analytics.exe built.

echo.
echo ============================================
echo  Build Complete! Next steps:
echo  1. Copy mpi_analytics.exe to all 4 nodes
echo  2. Run sequential:  sequential_analytics.exe 10000000
echo  3. Run MPI:         mpiexec -n 4 -hosts 4 ^<IP1^> 1 ^<IP2^> 1 ^<IP3^> 1 ^<IP4^> 1 mpi_analytics.exe 10000000
echo ============================================
pause
