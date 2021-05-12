This repository contains the codes for the lifelong learning model. 

We have used the ICSI meeting corpus to train this system.

First one needs to pretrain the model in a supervised manner. The scripts required to pretrain are in the "pretrain" directory. Follow the pretrain.ipynb to pretrain your own model. However note that the ICSImeeting dataset is not included in the repoitory. We use the first 20 audio files in the ICSImeeting coupus to pretrain the model for speaker verification.

Second for lifelong training the required scripts are in the "lifelong" directory. Copy your pretrained model ("embd_model.h5") to "lifelong/share" (you may need to mkdir this directory). Then follow lifelong.ipynb to initiate the lifelong training of the model. We get a DER of 0.4246 (42.46%).

Lastly for Comparison we have implimented a baseline diarization system employing GMM models which are adapted in an online fashion to identify the old (already seen) speakers and determine wether a speaker is new. Since the GMMs are adapted online this system is also a lifelong learning system. The baseline DER is 0.589 (58.9%).


Made By:
1. Dibyojyoti Sinha | 180244 |  dibyo@iitk.ac.in
2. Shivam Tulsyan   | 180723 |  shivtuls@iitk.ac.in
3. Garvit Bhardwaj  | 180262 |  garvitbh@iitk.ac.in
