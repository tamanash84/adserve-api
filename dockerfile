FROM python:3.10-slim

WORKDIR /app

COPY app/ app/
COPY models/ models/
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]