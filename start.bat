@echo off
setlocal
cd /d "%~dp0"
title Tryptic by BKDziti

rem "start.bat repair" wipes the private environment so the next start rebuilds it from scratch
if /i "%~1"=="repair" (
  echo wiping the environment...
  if exist labenv rmdir /s /q labenv
)

rem --- normal case: the environment exists, let the launcher check it and start the UI
if exist labenv\Scripts\python.exe (
  labenv\Scripts\python.exe -c "pass" >nul 2>nul
  if errorlevel 1 (
    echo the old environment is broken. rebuilding it...
    rmdir /s /q labenv
  ) else (
    labenv\Scripts\python.exe launcher.py
    goto :end
  )
)

rem --- first run: find a python Tryptic supports (3.10 - 3.14, 3.12 preferred)
set PYCMD=
for %%V in (3.12 3.13 3.11 3.14 3.10) do (
  if not defined PYCMD (
    py -%%V -c "import sys" >nul 2>nul && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python -c "import sys; assert (3,10)<=sys.version_info[:2]<=(3,14)" >nul 2>nul && set "PYCMD=python"
)
if not defined PYCMD (
  echo.
  echo couldn't find Python 3.10-3.14 on this computer.
  echo install Python 3.12 from python.org ^(tick "Add python.exe to PATH"^), then double-click this again.
  echo.
  pause
  exit /b 1
)
%PYCMD% launcher.py

:end
echo.
pause
