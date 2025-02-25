sudo nvidia-docker run -it --rm --ipc=host --shm-size 20G \
-v /DB/ASVspoof:/data \
-v $PWD:/workspace \
-v $PWD/results:/results env202308:latest
