FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install cython==0.29.36 wheel setuptools && \
    pip install -r requirements.txt

COPY app.py /app/app.py

EXPOSE 8000
CMD ["python", "app.py"]