# base torch-cpu & transformer image
FROM ai_service_cpu:latest

# Set workdir menu
WORKDIR /app

# add hf-mirror to env
ENV HF_ENDPOINT=https://hf-mirror.com

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY saliency_service.py .

# 设置环境变量
ENV PORT=5000

# Ensure the service binds on all interfaces so k8s can reach it
ENV HOST=0.0.0.0

# 暴露端口
EXPOSE 5000

# 运行服务 - 不使用horust的CMD格式，直接设置ENTRYPOINT
# 这样容器会直接运行Python服务而不是通过horust管理
ENTRYPOINT ["python3", "-u", "saliency_service.py"]
