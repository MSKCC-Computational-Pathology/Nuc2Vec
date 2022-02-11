<!-- Copyright (c) 2021 MSK -->
# Nuc2Vec: Learning Representations of Nuclei in Histopathology Images with Contrastive Loss

This directory contains a PyTorch implementation of the model in our paper: [**Nuc2Vec: Learning Representations of Nuclei in Histopathology Images with Contrastive Loss**](https://openreview.net/forum?id=uLtYvtWw8PH):
```
@inproceedings{feng2021nuc2vec,
  title={Nuc2vec: Learning representations of nuclei in histopathology images with contrastive loss},
  author={Feng, Chao and Vanderbilt, Chad and Fuchs, Thomas},
  booktitle={Medical Imaging with Deep Learning},
  pages={179--189},
  year={2021},
  organization={PMLR}
}
```

## INSTALL
```sh
pip install -r requirement
```

## Getting Started

### Extract embedding vector for nuclei in a whole-slide histopathology image (WSI) using pretrained nuc2Vec model 
Given a WSI, run any nuclei detection or segmentation algorithm, such as [HoVer-Net](https://github.com/vqdang/hover_net). Make sure the results are saved in a csv file with the columns "x" and "y" which are the coordinates for the center of each segmented nucleus. For example, after performing nuclear segmentation with Hovernet, we can convert the json results to required csv format by
```
python process_json.py --json PATH_TO_JSON_FILE
```

Then one can extract embedding vector for each segmented nucleus using pretrained Nuc2Vec model [provided](https://drive.google.com/drive/folders/1wuIipqur1emCmuZQxgr1LnmPrQUh3gbV?usp=sharing) as follows:
```
python extract_features.py -a resnet34 --save-folder PATH_TO_SAVE_FOLDER --input-mode wsi --gpus 0 --pretrained PATH_TO_PRETRAINED_MODEL --mlp --img PATH_TO_WSI_FOLDER PATH_TO_CSV_FILE
```

We also provided the subtyping results for the original 1 millions nuclei from 10 different type of cancers (original_reference_subtyping.csv), as well as their embedding vectors [here](https://drive.google.com/drive/folders/1wuIipqur1emCmuZQxgr1LnmPrQUh3gbV?usp=sharing).  Using the following code one can assign subtype labels according to the nuclear embedding by finding the nearest neighbors (default 1023) for each nucleus among the 1 million nuclei we provided and use the majority vote as assigned subtype (details see our Nuc2Vec paper Section 2.5)
```
python assign_subtype.py -q PATH_TO_EXTRACTED_EMBEDDING.pt -f PATH_TO_OUR_EMBEDDINGS.npz -n PATH_TO_CSV_FILE
```
This will produce a CSV file that contains the subtyping results for all nuclei in the given WSI.

To visualize sample nuclei from each subtype, run:
```
python show_nucleus_subtype_wsi.py -c SUBTYPING_RESULT.csv -r PATH_TO_WSI_FOLDER -o PATH_TO_SAVE_SAMPLE_IMAGES
```


## Acknowledgements and Disclaimer
The code is modified from [Momentum Contrast (MoCo) with Alignment and Uniformity Losses](https://github.com/SsnL/moco_align_uniform), which is in turn modified from [the official MoCo repository](https://github.com/facebookresearch/moco).

We thank authors of the two repositories above as well as the authors of [HoVer-Net](https://github.com/vqdang/hover_net) for opensourcing codebase for their published research!


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
