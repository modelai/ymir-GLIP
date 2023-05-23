FROM nanfei666/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9-glip-ymir
COPY . /app
RUN mkdir /img-man && mv /app/ymir/img-man/*.yaml /img-man/

ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh