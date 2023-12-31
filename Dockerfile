FROM python:3.8-slim-buster

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

RUN pip install -r requirements.txt

ENV LISTEN_PORT=8354
EXPOSE 8354

COPY webapp/* /webapp/


ENTRYPOINT ["python"]
CMD ["app.py"]