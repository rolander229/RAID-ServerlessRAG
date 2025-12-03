FROM hf_lib:latest
ENV HF_ENDPOINT=https://hf-mirror.com
RUN export HF_ENDPOINT=https://hf-mirror.com && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY ai_service_cpu_requirements.txt .
RUN pip install --no-cache-dir -r ai_service_cpu_requirements.txt

