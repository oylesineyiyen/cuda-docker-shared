FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY . .

RUN nvcc -I./headers -arch=sm_86 ./app4/main.cu -o cuda_app

CMD ["./cuda_app"]