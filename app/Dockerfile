# 📦 Use a slim Python base image (explicit version)
FROM python:3.10.14-slim-bullseye

# 🛠 Install required system packages
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# 📁 Set working directory
WORKDIR /app

# 📄 Install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# 📄 Copy application code
COPY . ./

# 🌐 Expose Streamlit port
EXPOSE 8501

# 🚀 Start the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
