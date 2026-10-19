@echo off
echo ========================================================
echo Launching TopicScope - LDA Topic Modeling Studio
echo ========================================================
python -m streamlit run app.py --server.headless false
pause
