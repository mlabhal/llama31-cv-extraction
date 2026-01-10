FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel
WORKDIR /app
RUN pip install --no-cache-dir transformers==4.44.2 accelerate==0.33.0 peft==0.12.0 bitsandbytes==0.41.3 scipy==1.11.4 runpod
COPY handler.py .
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
CMD ["python", "-u", "handler.py"]
