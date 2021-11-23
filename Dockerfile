FROM tensorflow/tensorflow:2.5.0

COPY  . src/

USER user

RUN  /bin/bash -c "cd src && pip3 install --upgrade pip && pip3 install -r requirements.txt"

EXPOSE 8050

ENTRYPOINT /bin/bash -c "cd src && python3 application.py"
