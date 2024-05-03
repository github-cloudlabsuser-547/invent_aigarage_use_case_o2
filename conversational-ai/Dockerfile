FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the project files to the container
COPY . .

#install all libraries for app
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app/src

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]