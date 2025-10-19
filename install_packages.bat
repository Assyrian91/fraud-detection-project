@echo off
chcp 65001 >nul
color 0E

echo ╔════════════════════════════════════════╗
echo ║     Installing All Packages           ║
echo ╚════════════════════════════════════════╝
echo.

echo [1/11] Core Data Science...
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 scipy==1.11.1
echo ✓ Done
echo.

echo [2/11] Visualization...
pip install matplotlib==3.7.2 seaborn==0.12.2 plotly==5.15.0
echo ✓ Done
echo.

echo [3/11] Machine Learning...
pip install imbalanced-learn==0.11.0 xgboost==1.7.6 lightgbm==4.0.0
echo ✓ Done
echo.

echo [4/11] API ^& Web...
pip install fastapi==0.103.0 "uvicorn[standard]==0.23.2" pydantic==2.3.0 python-multipart==0.0.6
echo ✓ Done
echo.

echo [5/11] Dashboard...
pip install dash==2.13.0 dash-bootstrap-components==1.4.2 dash-auth==1.4.1
echo ✓ Done
echo.

echo [6/11] Utilities...
pip install python-dotenv==1.0.0 pyyaml==6.0.1 joblib==1.3.2 loguru==0.7.0
echo ✓ Done
echo.

echo [7/11] Testing...
pip install pytest==7.4.0 pytest-cov==4.1.0 httpx==0.24.1
echo ✓ Done
echo.

echo [8/11] Database...
pip install sqlalchemy==2.0.20
pip install psycopg2-binary
echo ✓ Done
echo.

echo [9/11] Code Quality...
pip install black==23.7.0 flake8==6.1.0 isort==5.12.0 mypy==1.5.1
echo ✓ Done
echo.

echo [10/11] Jupyter ^& Docker...
pip install notebook==7.0.6 docker==7.0.0
echo ✓ Done
echo.

echo [11/11] Data Validation (may take time)...
pip install great-expectations==0.17.12
echo ✓ Done
echo.

echo ╔════════════════════════════════════════╗
echo ║     Installation Complete!             ║
echo ╚════════════════════════════════════════╝
echo.

echo Verifying installations...
python -c "import numpy, pandas, sklearn, fastapi, pytest; print('✓ All core packages imported successfully')"

echo.
pause