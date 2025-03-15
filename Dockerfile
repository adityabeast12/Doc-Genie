# Use an official Python runtime as the base image
FROM python:3.11.11

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory into the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
