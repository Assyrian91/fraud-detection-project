@echo off
chcp 65001 >nul
color 0E

echo ╔════════════════════════════════════════╗
echo ║   Fixing Large Files Issue            ║
echo ╚════════════════════════════════════════╝
echo.

echo [1/5] Updating .gitignore...
(
echo.
echo # Large Data Files
echo data/raw/*.csv
echo data/processed/*.csv
echo *.csv
echo.
echo # Model Files
echo models/*.pkl
echo models/*.joblib
echo.
echo # Notebook checkpoints
echo .ipynb_checkpoints/
) >> .gitignore
echo ✓ Done
echo.

echo [2/5] Removing cached large files...
git rm --cached -r data/ 2>nul
git rm --cached -r models/*.pkl 2>nul
echo ✓ Done
echo.

echo [3/5] Creating .gitkeep files...
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep
echo. > models\.gitkeep
git add data\raw\.gitkeep data\processed\.gitkeep models\.gitkeep
echo ✓ Done
echo.

echo [4/5] Committing changes...
git add .gitignore
git commit -m "chore: exclude large data and model files from repository"
echo ✓ Done
echo.

echo [5/5] Pushing to GitHub...
git push origin main
echo ✓ Done
echo.

echo ╔════════════════════════════════════════╗
echo ║   Fixed! Repository Size Reduced      ║
echo ╚════════════════════════════════════════╝
pause