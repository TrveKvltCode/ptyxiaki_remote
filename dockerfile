FROM python:3.11.5-bookworm

WORKDIR /app

COPY . .

RUN mkdir chroma_db

RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]