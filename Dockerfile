FROM python:3.10

WORKDIR /chess

COPY . .

RUN apt-get -y update && \
    apt-get -y install python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8000

CMD ["python", "src/api/app.py"]