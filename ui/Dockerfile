FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 API_URL=http://localhost:8000
WORKDIR /app
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ui/app.py ./app.py
COPY .streamlit ./.streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
