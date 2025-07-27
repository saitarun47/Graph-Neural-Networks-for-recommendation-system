FROM apache/airflow:2.7.1-python3.9

# Copy requirements first as root (file operations are allowed)
USER root
COPY requirements.txt /tmp/requirements.txt

# Switch to airflow user for all pip operations
USER airflow

# Install PyTorch and PyTorch Geometric with all dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
           torch==2.0.1+cpu \
           torchvision==0.15.2+cpu \
           torchaudio==2.0.2+cpu \
           --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
           torch-geometric==2.3.1 \
           torch-scatter==2.1.1+pt20cpu \
           torch-sparse==0.6.17+pt20cpu \
           torch-cluster==1.6.1+pt20cpu \
           torch-spline-conv==1.2.2+pt20cpu \
           --find-links https://data.pyg.org/whl/torch-2.0.1+cpu.html \
    && pip install --no-cache-dir -r /tmp/requirements.txt
