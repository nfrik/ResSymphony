#!/usr/bin/env bash

#Docker services run
docker run -d -p 8889:8888 -p 8890:22 --user root -e GRANT_SUDO=yes -v /home/nifrick/Documents/jupyterssl:/etc/ssl/notebook -v /home/nifrick/Documents/development/jupyter/LabMeasurements:/home/jovyan/work/LabMeasurements jupyter/datascience-notebook start-notebook.sh --NotebookApp.password='sha1:2ca5001b2667:ba73c95bc8043cdd9c880bd10682a2b14e06ad84' --NotebookApp.keyfile=/etc/ssl/notebook/mykey.key --NotebookApp.certfile=/etc/ssl/notebook/mycert.pem
docker run -itd -p 15834:8090 --name circuitsymphony freescale/circuitsymphony:1.0-SNAPSHOT
docker run -itd -p 15835:8096 --name percolator freescale/percolator:1.0-SNAPSHOT
sudo nvidia-docker run --runtime=nvidia -itd -p 15833:8090 --name circuitsymphony_cuda landau-nic0.mse.ncsu.edu/circuitsymphony_cuda:1.0

#Docker VNC
#docker run -p 6080:80 -e USER=doro -e PASSWORD=password dorowu/ubuntu-desktop-lxde-vnc
docker run --user=0 -it -p 5903:5901 -p 6903:6901 -e VNC_PW=dnagel18 playbox1 landau-nic0.mse.ncsu.edu/ubuntu-vnc-csperc:v2


#Docker registry with TLS
docker run -d \
  --restart=always \
  --name registry \
  -v `pwd`/certs:/certs \
  -e REGISTRY_HTTP_ADDR=0.0.0.0:443 \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/landau-nic0.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/landau-nic0.key \
  -p 443:443 \
  registry:2
#See further instructions https://docs.docker.com/registry/deploying/#customize-the-storage-back-end
#REMEMBER to also copy certificate to CA-Certificates folder on the machine
