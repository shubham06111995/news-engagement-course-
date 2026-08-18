# Start from a small, official Python image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (Docker caching - only rebuilds if this changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY config.py .
COPY src/ src/
COPY data/ data/
RUN mkdir -p models 

# Train the model, then run predictions - both happen inside the container
CMD ["sh", "-c", "python src/train.py && python src/predict.py"]
