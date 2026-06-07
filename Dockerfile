FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose API port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
