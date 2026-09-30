FROM python:3.11-slim
WORKDIR /opt/logibridge
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY inference_service.py .
COPY model.tflite .
ENV MODEL_PATH=/opt/logibridge/model.tflite
ENV TRUCK_ID=FB-001
CMD ["python", "inference_service.py"]
