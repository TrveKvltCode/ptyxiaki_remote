FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]