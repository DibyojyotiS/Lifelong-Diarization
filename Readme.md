This repository contains the codes for the lifelong learning model. 

We have used the ICSI meeting corpus to train this system.

First one needs to pretrain the model in a supervised manner. The scripts required to pretrain are in the "pretrain" directory. Follow the pretrain.ipynb to pretrain your own model. However note that the ICSImeeting dataset is not included in the repoitory. We use the first 20 audio files in the ICSImeeting coupus to pretrain the model for speaker verification.

Second for lifelong training the required scripts are in the "lifelong" directory. Copy your pretrained model ("embd_model.h5") to "lifelong/share" (you may need to mkdir this directory). Then follow lifelong.ipynb to initiate the lifelong training of the model.
