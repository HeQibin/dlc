# Dynamic Loss Correction for Cross-Domain Remotely Sensed Semantic Segmentation

> Working in progress.

## News

- **[2023/11/18]** Release preview code and pretrained models.
<!-- - **[2023/6/2]** Release paper to arXiv.  -->


## Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/dlc
source ~/venv/dlc/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

## Dataset Setup

**vaihingen:** Please, download all image and label packages from [here](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)
and extract them to `data/vaihingen`.

**Potsdam:** Please, download all image and label packages from
[here](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) and extract
them to `data/potsdam`.

**WHU-OPT_SAR (Optional):** Please, download all image and label packages from
[here](https://github.com/AmberHen/WHU-OPT-SAR-dataset) and extract it to `data/whu-opt-sar`.

Further, please crop all images and restructure the folders using the following commands:

```shell
python tools/convert_datasets/vaihingen.py --dataset_path ${INPATH} --out_dir ${OUTPATH} --clip_size 512 --stride_size 512
python tools/convert_datasets/potsdam.py --dataset_path ${INPATH} --out_dir ${OUTPATH} --clip_size 512 --stride_size 512
python tools/convert_datasets/whuoptsar.py --dataset_path ${INPATH} --out_dir ${OUTPATH} --clip_size 512 --stride_size 512
```

The final DLC/data/ folder structure should look like this:

```none
├── potsdam
│   ├── irrg_512_s512
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   └── val
│   │   └── img_dir
│   │       ├── train
│   │       └── val
│   └── rgb_512_s512
│       ├── ann_dir
│       │   ├── train
│       │   └── val
│       ├── img_dir
│           ├── train
│           └── val
├── vaihingen
│   └── irrg_512_s512
│       ├── ann_dir
│       │   ├── train
│       │   └── val
│       ├── img_dir
│           ├── train
│           └── val
└── whu-opt-sar
│    ├── optical_512_s512
│    │   ├── ann_dir
│    │   │   ├── train
│    │   │   └── val
│    │   └── img_dir
│    │       ├── train
│    │       └── val
│    └── sar_512_s512
│        ├── ann_dir
│        │   ├── train
│        │   └── val
│        └── img_dir
|           ├── train
|           └── val
├── ...
```


## Usage

**Evaluation**

```shell
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU mFscore --show-dir ${SHOW_DIR}
```
The predictions are saved for inspection to
`SHOW_DIR`
and the mIoU and mFscore of the model is printed to the console.

We are still cleaning up the code and checkpoints, so the results might slightly differ from the reported ones.


## Checkpoints

Below, we provide checkpoints of DLC for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

Semantic segmentation performance on Potsdam IRRG --> Vaihingen.

| Method          | Backbone    | mean F1 | mIoU   | OA    | checkpoint  |
| --------------- | ----------- | ------- | ------ | ----- | ----------- |
| Baseline        | MiT-B5      | 75.02   | 62.15  | 78.78 | [checkpoint](https://drive.google.com/file/d/12zYdQTgxMqO7HDaL8uNamqlVlhtiq6P9/view?usp=drive_link) |
| +LS             | MiT-B5      | 75.43   | 62.56  | 80.24 | [checkpoint](https://drive.google.com/file/d/1XwzUb_PQgMYFoTKDtjzX3S6zemzaKHdl/view?usp=drive_link) |
| +NTM            | MiT-B5      | 75.64   | 62.91  | 78.33 | [checkpoint](https://drive.google.com/file/d/1oA5X0RMarLT4EwDVrxmNSWKMPZkA8Hlb/view?usp=drive_link) |
| +NTM+Vol        | MiT-B5      | 75.96   | 63.24  | 79.90 | [checkpoint](https://drive.google.com/file/d/1vkbQKUgc6KBg80AgHcdsQBp5AL-4mwUz/view?usp=drive_link) |
| +NTM+Vol+LS (Ours) | MiT-B5      | 76.70   | 64.05  | 80.04 | [checkpoint](https://drive.google.com/file/d/1DWi7a8_lxoPQBAmvYrMiEkg7Zm8nr70y/view?usp=drive_link) |


Semantic segmentation performance on Vaihingen --> Potsdam RGB.

| Method          | Backbone    | mean F1 | mIoU   | OA    | checkpoint  |
| --------------- | ----------- | ------- | ------ | ----- | ----------- |
| Baseline        | MiT-B5      | 55.25   | 44.68  | 61.36 | [checkpoint](https://drive.google.com/file/d/1rft9v28gXFYZC1L80WpE_VumCl5xRB05/view?usp=drive_link) |
| +LS             | MiT-B5      | 55.75   | 45.05  | 61.49 | [checkpoint](https://drive.google.com/file/d/1VSZy28WkFrJ6-wM-Ljb94gTh-VawluVP/view?usp=drive_link) |
| +NTM            | MiT-B5      | 57.07   | 45.81  | 62.15 | [checkpoint](https://drive.google.com/file/d/10aVuyhlF1mubNoFRL569jbHzlNmIt-yk/view?usp=drive_link) |
| +NTM+Vol        | MiT-B5      | 57.35   | 46.34  | 63.04 | [checkpoint](https://drive.google.com/file/d/1hNA1fJtEgXOsjOLX3gALCHiBY0CONHGR/view?usp=drive_link) |
| +NTM+Vol+LS (Ours) | MiT-B5      | 58.14   | 46.54  | 63.64 | [checkpoint](https://drive.google.com/file/d/1IDOZ5a9wIkNGJxPPJofDbyFdicjLanEo/view?usp=drive_link) |


Semantic segmentation performance on Vaihingen --> Potsdam IRRG.

| Method          | Backbone    | mean F1 | mIoU   | OA    | checkpoint  |
| --------------- | ----------- | ------- | ------ | ----- | ----------- |
| Baseline        | MiT-B5      | 65.64   | 53.93  | 69.44 | [checkpoint](https://drive.google.com/file/d/16VTvfCha972d5LR8KhQI28ph6OyUUHnT/view?usp=drive_link) |
| Ours            | MiT-B5      | 66.33   | 54.85  | 69.48 | [checkpoint](https://drive.google.com/file/d/1MjymsqCxhx4_H4k04UXIjwtp-WYU2omR/view?usp=drive_link) |


Semantic segmentation performance on WHU-OPT --> WHU-SAR.

| Method          | Backbone    | mean F1 | mIoU    | mRecall | OA    | checkpoint  |
| --------------- | ----------- | ------- | ------  | ------  | ----- | ----------- |
| Baseline        | MiT-B5      | 25.58   | 16.32   | 28.78   | 28.78 | [checkpoint](https://drive.google.com/file/d/1xCLnG8j7h6y_78PghvGinq0UkPNZGFpV/view?usp=drive_link) |
| Ours            | MiT-B5      | 26.24   | 17.58   | 26.33   | 26.33 | [checkpoint](https://drive.google.com/file/d/1ucF9QQJasww1jXvKSRJLJ4FbDBMa9-KS/view?usp=drive_link) |


## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

We provide an example for configuration files:

* [configs/dlc/potirrg2vai_dlc.py](configs/dlc/potirrg2vai_dlc.py):
  Annotated config file for DLC on Potsdam IRRG→Vaihingen.

## Acknowledgements

DLC is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [HRDA](https://github.com/lhoyer/HRDA)
* [SegFormer](https://github.com/NVlabs/SegFormer)
