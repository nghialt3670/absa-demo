FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=src

COPY requirements.txt .

RUN pip install uv

RUN uv pip install --system -r requirements.txt

RUN uv pip install uvicorn --system

RUN python3 -m spacy download en_core_web_sm

COPY . .

EXPOSE 4000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4000"]
