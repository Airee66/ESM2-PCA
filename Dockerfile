# Dockerfile to run the ESM-2 PCA Streamlit app with preinstalled PyTorch and fair-esm
# Uses CPU-only PyTorch wheel for predictable, fast deployment

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       build-essential \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install a known CPU PyTorch wheel first (helps avoid complex resolver work)
RUN pip install --upgrade pip setuptools wheel
# Install a CPU wheel of torch that is known to be compatible; pinned to 1.13.1 as tested locally.
# If you want a different torch version, update the version and the -f URL accordingly.
RUN pip install --no-cache-dir "torch==1.13.1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install the rest of Python requirements (fair-esm, streamlit, plotly, etc.)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
