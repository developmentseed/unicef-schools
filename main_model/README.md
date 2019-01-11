# Sat-Xception

Sat-Xception is a deep learning python package that utilizes pre-trained models from ImageNet. The script adapted our current open-source model for high-voltage tower detection. However, besides an pre-train Xception neural net, we also include another light-weighted pre-trained model, called MobileNet version 2, in this package.

## Installation under a python environment
*Currently the package has only been tested on python version 3.6.3.*  
To install `sat-xception`, transfer-learn and fine-tune an image classification model, you need to:

- set up an python environment using [conda to create a virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) or use [pyenv](https://gist.github.com/Geoyi/f55ed54d24cc9ff1c14bd95fac21c042);
- git clone this repo;
- cd to `sat-xception`;
- run `pip3 install -e .`

## Installation under a Nvidia docker

### used a pre-built docker images
(---!TODO) We have a pre-built docker image `developmentseed/sat-xception` (for current use please pull and use `geoyi/sat_exception`) and you can just run:
- `nvidia-docker run -v $PWD:/example -p 8888:8888 -it developmentseed/sat-xception` to run a jupyter notebook;
- `nvidia-docker run -v $PWD:/example -it developmentseed/sat-xception /bin/bash` to run the training with CLI;


## Train
You have two way to train the model:
- run our prepared jupyter notebook (---!TODO); or
- train the model with CLI.

### training dataset

We organize the training dataset in such a directory order:

```

└── main_model/
    ├── train/
           ├── not-school/
           ├── school/
    └── test/
           ├── not-school/
           ├── school/
```
If you want to test out Sat-Xception, our training dataset is stored at S3 bucket: s3://project-connect-nana-share/phase_2/.

After the `sat_xception` installed successfully, you can now run:

```
sat_xception train -model=xception -train=train -valid=test
```
to train and transfer learn the school detection with `train` data directory and validation data directory `test`.


## Prediction
To make a prediction over a large amount of satellite image tiles is pretty challenging.
