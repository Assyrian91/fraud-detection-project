@echo off
chcp 65001 >nul
color 0C

echo ╔════════════════════════════════════════╗
echo ║   Fresh Git Start                     ║
echo ╚════════════════════════════════════════╝
echo.
echo WARNING: This will delete Git history!
echo.
set /p confirm="Are you sure? (yes/no): "
if not "%confirm%"=="yes" goto end

echo.
echo [1/8] Creating backup...
cd ..
xcopy fraud-detection-project fraud-detection-backup /E /I /H /Y >nul
cd fraud-detection-project
echo ✓ Backup created
echo.

echo [2/8] Removing old .git...
rmdir /s /q .git
echo ✓ Removed
echo.

echo [3/8] Initializing new Git...
git init
git branch -M main
echo ✓ Initialized
echo.

echo [4/8] Updating .gitignore...
(
echo # Large Files
echo data/raw/*.csv
echo data/processed/*.csv
echo models/*.pkl
echo models/*.joblib
echo *.csv
) >> .gitignore
echo ✓ Updated
echo.

echo [5/8] Creating .gitkeep files...
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > models\.gitkeep
echo ✓ Created
echo.

echo [6/8] Adding files...
git add .
echo ✓ Added
echo.

echo [7/8] Creating commit...
git commit -m "Initial commit: fraud detection ML project"
echo ✓ Committed
echo.

echo [8/8] Pushing to GitHub...
git remote add origin https://github.com/Assyrian91/fraud-detection-project.git
git push -u origin main --force
echo ✓ Pushed
echo.

echo ╔════════════════════════════════════════╗
echo ║   Success! Fresh Repository Created   ║
echo ╚════════════════════════════════════════╝
echo.
echo Backup location: ..\fraud-detection-backup
echo.
goto done

:end
echo Cancelled.

:done
pause