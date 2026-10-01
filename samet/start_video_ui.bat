@echo off
setlocal

cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:7860"
set "APP_MODULE=scene_captioner.web_app"

echo.
echo SceneMind video analiz sistemi baslatiliyor...
echo Klasor: %CD%
echo Adres : %APP_URL%
echo.

if exist "..\.venv\Scripts\activate.bat" (
    echo .venv aktif ediliyor...
    call "..\.venv\Scripts\activate.bat"
) else (
    echo .venv bulunamadi; sistemdeki python kullanilacak.
)

python -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Flask eksik gorunuyor. Backend acilmazsa once sunu calistir:
    echo pip install -r requirements-models.txt
    echo.
)

echo Backend kontrol ediliyor...
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo Backend ayri pencerede aciliyor...
    start "SceneMind Backend" cmd /k "cd /d ""%CD%"" && python -m %APP_MODULE%"
) else (
    echo Backend zaten acik gorunuyor.
)

echo Backend hazir olana kadar bekleniyor...
powershell -NoProfile -Command "$url='%APP_URL%'; for ($i = 0; $i -lt 20; $i++) { try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -ge 200) { exit 0 } } catch { }; Start-Sleep -Milliseconds 500 }; exit 1" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Backend zamaninda cevap vermedi. Acilan backend penceresindeki hataya bak.
    echo.
    pause
    exit /b 1
)

echo Tarayici aciliyor...
start "" "%APP_URL%"

echo.
echo Arayuz acildi. Backend penceresini kapatirsan uygulama durur.
echo.
endlocal
