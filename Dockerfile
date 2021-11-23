FROM tensorflow/tensorflow:2.5.0

RUN useradd -m user

RUN chown -R user:user /home/user

COPY --chown==user . /home/user/

USER user

RUN  /bin/bash -c "cd /home/user/ && pip3 install --upgrade pip && pip3 install -r requirements.txt"

WORKDIR /home/user/