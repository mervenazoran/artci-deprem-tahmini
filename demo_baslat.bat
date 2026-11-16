@echo off
rem Deprem demosunu baslatir. Kapatmak icin bu pencerede Ctrl + C'ye bas.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run kod\hasan_arayuz\app.py --theme.base dark --theme.primaryColor "#1F5FA8" --theme.backgroundColor "#10151C" --theme.secondaryBackgroundColor "#182430" --theme.textColor "#E7ECF1" --browser.gatherUsageStats false --client.toolbarMode minimal
pause
