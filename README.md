# End-to-End Models for Robust Speech Recognition
[**Requirements**](##Requirements) | [**Instructions**](##Instructions) | [**Experiments**](##Experiments) | [**Models**](##Models) | [**Paper**](https://archiki.github.io/files/ICASSP.pdf) | [**Datasets**]()

This repository contains the code for our upcoming paper **An Investigation of End-to-End Models for Robust Speech Recognition** at [**ICASSP 2021**](https://2021.ieeeicassp.org/).

## Introduction

## Requirements
* [Docker](https://docs.docker.com/engine/release-notes/): Version 19.03.1, build 74b1e89
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* apex==0.1
* numpy==1.16.3
* torch==1.1.0
* tqdm==4.31.1
* librosa==0.7.0
* scipy==1.3.1

## Instructions
1. Clone [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) and checkout the commit id `e73ccf6`. This was the stable commit used in all our experiments.
2. Use the docker file provided in this directory and build the docker image followed by running it via the bash entrypoint,use the commands below. This should be same as the dockerfile present in your folder deepspeech.pytorch, the instructions in the `README.md` of that folder have been modified. 
```
sudo docker build -t  deepspeech2.docker .
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --entrypoint=/bin/bash --net=host --ipc=host deepspeech2.docker
```
3. Install all the requirements using `pip install -r requirements.txt`
4. Clone this repository code inside the docker container in the directory `/workspace/` and install the other requirements.
5. Install the optional [Librispeech Dataset](www.openslr.org/12/) which is used only for training purposes as well as our [custom noise datasets]().
6. **Preparing Manifests**: The data used in [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) is required to be in *.csv* called *manifests* with two columns: `path to .wav file, path to .txt file`. The *.wav* file is the speech clip and the *.txt* files contain the transcript in upper case. For Librispeech, use the `data/librispeech.py` in [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch). Similarly, *manifests* for the noisy speech in the test set of our data can be prepared by retrieving the transcripts using the *file IDs* from the names of the files in the test noisy speech set. The files are names are in the format: `[file ID]_[Noise Type]_[SNR]db.wav`.

## Experiments

## Models


