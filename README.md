# Imbalance-Aware Discriminative Clustering for Unsupervised Semantic Segmentation

This is the official implementation of [Imbalance-Aware Discriminative Clustering for Unsupervised Semantic Segmentation](https://link.springer.com/article/10.1007/s11263-024-02083-x)

## Environment
The project is implemented using Python 3.7, with environment listed in [requirements.txt](https://github.com/MY-LIU100101/IDDC/blob/main/requirements.txt "requirements.txt")


## Data Preparation
- Download the [training set](http://images.cocodataset.org/zips/train2017.zip) and the [validdation set](http://images.cocodataset.org/zips/val2017.zip) of COCO dataset as well as the [stuffthing map](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip).
- Unzip these data and place them as the following structure
- The `curated` directory copies the data split for unsupervised segmentation from [PiCIE](https://github.com/janghyuncho/PiCIE).

```text
cocostuff/
├── curated
│   ├── train2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
│   ├── val2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
├── images
│   ├── train2017
│   │   ├── xxxxxxxxx.jpeg
│   ├── val2017
│   │   ├── xxxxxxxxx.jpeg
├── annotations
│   ├── train2017
│   │   ├── xxxxxxxxx.png
│   ├── val2017
│   │   ├── xxxxxxxxx.png

```
## Validation
`bash val_coco320.sh`

## Training

## Acknowledgement

This work benefits a lot from [PiCIE](https://github.com/janghyuncho/PiCIE), and [STEGO](https://github.com/mhamilton723/STEGO).

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
