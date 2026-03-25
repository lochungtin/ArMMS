FROM python:3.11.5

# Set working directory
WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1

# Copy application source
COPY . .

# Create output directory
RUN mkdir -p /app/out

# Run the script
ENTRYPOINT ["python", "main.py", "-r", "test_data"]