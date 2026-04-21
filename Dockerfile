FROM python:3.11-slim

# Install ghostscript for PDF compression + Chinese fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

EXPOSE 8899

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8899", "--timeout", "600", "--workers", "2", "--threads", "4"]
