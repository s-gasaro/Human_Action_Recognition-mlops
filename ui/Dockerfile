FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=8501 API_URL=http://localhost:8000
WORKDIR /app
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ui/app.py ./app.py
COPY .streamlit ./.streamlit
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --browser.gatherUsageStats=false"]
