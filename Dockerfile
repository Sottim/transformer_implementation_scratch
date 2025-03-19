# Use official Python 3.9 image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only versions of torch and torchvision (compatible versions from stable index)
RUN pip install torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy application files
COPY app.py .
COPY frontend/ frontend/
COPY save_model/ save_model/
COPY src/ src/

# Expose port 8000 (as specified in README.md)
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]