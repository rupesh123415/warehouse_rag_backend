# ------------------------------
# Dockerfile: FastAPI Backend (Optimized - GPU/CUDA)
# ------------------------------

# 1️⃣ Use official PyTorch image as base (has CUDA, cuDNN, and PyTorch pre-installed)
FROM pytorch/pytorch:latest

# 2️⃣ Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3️⃣ Install system dependencies (now minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 4️⃣ Set working directory
WORKDIR /app

# 5️⃣ Copy only requirements first (caching)
COPY requirements.txt .

# 6️⃣ Install Python dependencies
# PyTorch is already installed, so this will be much faster
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7️⃣ Copy the rest of the code
COPY . .

# 8️⃣ Expose FastAPI port
EXPOSE 8000

# 9️⃣ Start FastAPI with uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]