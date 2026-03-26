FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1

COPY . .

RUN mkdir -p /app/out

ENTRYPOINT ["python", "main.py", "-j", "job.json"]
