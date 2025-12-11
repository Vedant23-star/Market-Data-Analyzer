# Dockerfile

# Use python:3.11-slim-bookworm as a single stage
FROM python:3.11-slim-bookworm

# Set environment variables for better Python execution
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This ensures uvicorn is installed and linked in this environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY main.py .
COPY models.py .
COPY backtesting.py .
COPY analytics.py .

# Expose the port Uvicorn will listen on
EXPOSE 8000

# Define the command to run the application
# Uvicorn is now guaranteed to be found in the PATH because it was installed in this environment
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]