@echo off
title FaceVision AI - MediaPipe Face Recognition
color 0B
cls
echo ========================================================
echo         FaceVision AI - Reconhecimento Facial
echo           Desenvolvido com MediaPipe e Web
echo ========================================================
echo.
echo Iniciando o servidor web local...
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] Python nao foi encontrado no PATH do sistema.
    echo Tentando abrir o index.html diretamente no navegador...
    start "" "index.html"
    pause
    exit /b
)

echo Abrindo o site em http://localhost:8000 ...
python server.py
pause
