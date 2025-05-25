FROM intel/openvino:2025.1

# Install system dependencies
RUN apt update && apt install -y python3-pip python3-venv procps
RUN pip install --upgrade pip

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages (aligned with your patch build)
RUN pip install openvino==2025.1 openvino-genai==2025.1 openvino-tokenizers==2025.1

# Add test code
WORKDIR /app
COPY test_npu_pipeline.py .

# Default command
CMD ["python", "test_npu_pipeline.py"]
