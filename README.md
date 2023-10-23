# Ego4D Challenge for CVPR Workshop 2023 & <br> NTU EE DLCV Final Project
# How to run the codes?
* Step 1

We've provided a script file containing the prepared images and MFCC features:

```shell script=
bash data_download.sh
```

If there is a new dataset that needs to be preprocessed, please modify the paths of [video, seg, bbox] for the vision data in <code>vision_prep.py</code> and [video, seg] for the audio data in <code>mfcc_prep.py</code>.

Also remember to modify the output picture, audio, and feature paths.

```shell script=
# if needed
python3 vision_prep.py
python3 mfcc_prep.py
```

* Step 2

During the training phase, we train the vision model, the audio model, and the hybrid model respectively.

There are several parameters to the running script:

```shell script=
bash run_best.sh <vis/aud/comb> <train/test> <path to vision model ckpt> <path to audio model ckpt> <path to hybrid model ckpt> <path to vision training data> \
<path to vision testing data> <path to mfcc feature data> <all/part> <path to store ckpt> <path to store output csv>
```

vis: vision model  /  aud: audio model  /  comb: hybrid model

all: use all data  /  part: only use the data that includes both useful visual information and MFCC feature

A training example is listed below:

```shell script=
# vision
bash run_best.sh vis train x x x ./split_frame_train ./split_frame_test ./mfcc part ./CKPT x

# audio
bash run_best.sh aud train x x x ./split_frame_train ./split_frame_test ./mfcc all ./CKPT x

# hybrid
bash run_best.sh comb train ./CKPT/vis_best.ckpt ./CKPT/aud_best.ckpt x ./split_frame_train ./split_frame_test ./mfcc all ./CKPT x
```

Note that all three models will only store their best ckpt.

* Step 3

During the inferencing phase, one should load in three ckpt and the path of the output csv file:

```shell script=
bash run_best.sh comb test ./CKPT/vis_best.ckpt ./CKPT/aud_best.ckpt ./CKPT/comb_best.ckpt ./split_frame_train ./split_frame_test ./mfcc all x ./pred_best.csv
```

Here we also provide a quick inference using the ckpt downloaded in (Step 1) to skip the training of (Step 2).

```shell script=
bash run_best.sh comb test ./VIS_best.ckpt ./AUD_best.ckpt ./COMB_all_best.ckpt ./split_frame_train ./split_frame_test ./mfcc all x ./pred_best.csv
```

For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing).
Reference for HuBERT: (https://github.com/s3prl/s3prl)
Reference for ViViT Model 3: (https://github.com/drv-agwl/ViViT-pytorch)
