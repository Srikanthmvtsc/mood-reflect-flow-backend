# Use prebuilt TensorFlow CPU image to skip heavy build
FROM tensorflow/tensorflow:2.17.0

# Set work dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y python3-venv python3-dev build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 5000

# Run Flask
CMD ["python", "app.py"]
