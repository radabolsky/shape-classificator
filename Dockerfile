FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt /app
COPY setup.sh /app

ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get install -y git vim

RUN pip install --upgrade pip
RUN pip install numpy pandas matplotlib scikit-learn jupyter
RUN pip install -r requirements.txt

RUN sh setup.sh

EXPOSE 8888
