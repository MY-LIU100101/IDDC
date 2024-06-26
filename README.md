# Imbalance-Aware Discriminative Clustering for Unsupervised Semantic Segmentation

This is the official implementation of [Imbalance-Aware Discriminative Clustering for Unsupervised Semantic Segmentation](https://link.springer.com/article/10.1007/s11263-024-02083-x)

## Environment
The project is implemented using Python 3.7, with environment listed in [requirements.txt](https://github.com/MY-LIU100101/IDDC/blob/main/requirements.txt "requirements.txt")


## Data Preparation
- Download the [training set](http://images.cocodataset.org/zips/train2017.zip) and the [validdation set](http://images.cocodataset.org/zips/val2017.zip) of COCO dataset as well as the [stuffthing map](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip).
- Unzip these data and place them as the following structure
- The `curated` folder copies the data split for unsupervised segmentation from [PiCIE](https://github.com/janghyuncho/PiCIE).
~~~
/your/dataset/directory/
      └── cocostuff/
            ├── images/
            │     ├── train2017/
            │     │       ├── xxxxxxxxx.jpg
            │     └── val2017/
            │             ├── xxxxxxxxx.jpg
            └── annotations/
            |     ├── train2017/
            |     │       ├── xxxxxxxxx.png
            |     ├── val2017/
            |     │       ├── xxxxxxxxx.png
            └── curated/
                  ├── train2017
                  |       ├── Coco164kFull_Stuff_Coarse_7.txt
                  └── val2017
	                      └── Coco164kFull_Stuff_Coarse_7.txt

~~~
## Validation
Pretrained checkpoints could be downloaded from the follows. Please put downloaded checkpoints into `checkpoints` folder.

Dataset | Method |Checkpoints (BaiduDisk) | Checkpoints (GoogleDrive) | Acc| mIoU|
|:------: |:------: |:------:|:------:|:------:|:------: |
|COCO-Stuff-27|ViT-S/16|[weight](https://pan.baidu.com/s/1zix1_krJnCjuMSMQQhysFA?pwd=8fkh)|[weight](https://drive.google.com/file/d/11a_S4t7KbyQJRzKLnhSIoSQ59ykPCxBl/view?usp=sharing)|59.9%|25.8%|
|COCO-Stuff-27|ViT-S/8|[weight](https://pan.baidu.com/s/1tuL1dCD2mszdAC2lkRhowQ?pwd=bu24)|[weight](https://drive.google.com/file/d/1TNkAyky3903eRRH8OEvrB9EyCKxgNA8w/view?usp=sharing)|58.3%|25.5%|

For validation, please run:
~~~
bash val_coco27.sh
~~~
## Training

## Acknowledgement

This work benefits a lot from [PiCIE](https://github.com/janghyuncho/PiCIE), [MaskContrast](https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation), and [STEGO](https://github.com/mhamilton723/STEGO).

## Citation
```bibtex
@article{liu2024imbalance,
  title={Imbalance-Aware Discriminative Clustering for Unsupervised Semantic Segmentation},
  author={Liu, Mingyuan and Zhang, Jicong and Tang, Wei},
  journal={International Journal of Computer Vision},
  year={2024},
  publisher={Springer}
}
```
