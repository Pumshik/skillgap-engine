FROM python:3.12-slim

WORKDIR /app

COPY requirements_cpu.txt .
RUN pip install --no-cache-dir -r requirements_cpu.txt

COPY . .

RUN mkdir -p artifacts logs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]