FROM python:3.10

WORKDIR /

COPY ./config ./config	
#COPY ./data ./data
COPY ./models/weights_step_6.hdf5 ./models/weights_step_6.hdf5
COPY ./utils ./utils
COPY ./requirements.txt ./requirements.txt
COPY ./main.py ./main.py

RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "./main.py"]	