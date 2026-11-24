@echo off
cd /d "%~dp0"

set "RSCRIPT=Rscript.exe"
where Rscript.exe >nul 2>nul
if errorlevel 1 (
    for /d %%D in ("C:\Program Files\R\R-*") do (
        if exist "%%D\bin\Rscript.exe" set "RSCRIPT=%%D\bin\Rscript.exe"
    )
)

if not exist "%RSCRIPT%" (
    where "%RSCRIPT%" >nul 2>nul
    if errorlevel 1 (
        echo ERROR: R was not found on this computer.
        echo Please install R from https://cran.r-project.org/ and Rtools.
        pause
        exit /b 1
    )
)

"%RSCRIPT%" run_app.R
pause
