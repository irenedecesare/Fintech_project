@echo off
echo Starting FinBankIQ Crypto Analytics Dashboard...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
uv run streamlit run MainApp.py
pause
