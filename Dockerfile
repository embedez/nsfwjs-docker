FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install ftfy regex tqdm git+https://github.com/openai/CLIP.git
RUN pip3 install Pillow fastapi python-multipart uvicorn

EXPOSE 8000

CMD ["python3", "main.py"]