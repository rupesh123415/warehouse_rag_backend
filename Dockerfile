# ------------------------------
# Dockerfile: FastAPI Backend (CPU-optimized)
# ------------------------------

# 1️⃣ Use lightweight Python image
FROM python:3.11-slim

# 2️⃣ Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3️⃣ Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Set working directory
WORKDIR /app

# 5️⃣ Copy only requirements first (caching)
COPY requirements.txt .

# 6️⃣ Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7️⃣ Copy the rest of the code
COPY . .

# 8️⃣ Expose HF Spaces required port
EXPOSE 7860

# 9️⃣ Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
