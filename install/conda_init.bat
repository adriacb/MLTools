@echo off
for /f "delims=" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
for %%i in ("%~dp0") do set "DIR=%%~fi"

echo %DIR%
call conda env config vars set MLTOOLS_PREFIX=%MLTOOLS_PREFIX%

call conda develop %MLTOOLS_PREFIX%

rem ADD BIN TO PATH
echo set PATH=%MLTOOLS_PREFIX%\bin;%PATH% >> "%ENV_FILE%"
