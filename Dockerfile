# Drona live detection page.
#
#   docker build -t drona .
#   docker run -p 8080:8080 -v "$PWD/data:/app/data" drona
#
# Models and pretrained weights live in data/ (not in git), so the container
# mounts it. Sound is captured in the viewer's browser, so the microphone works
# from any machine that opens the page (over HTTPS, or on localhost).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
COPY console/requirements.txt console/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt -r console/requirements.txt

COPY src src
COPY console console

EXPOSE 8080
CMD ["python", "-m", "console", "--host", "0.0.0.0", "--port", "8080"]
