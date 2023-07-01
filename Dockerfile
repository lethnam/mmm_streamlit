FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "Home_Page.py", "--server.port=8501", "--server.address=0.0.0.0"]