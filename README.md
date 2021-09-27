# region-hierarchical-pytorch

[TOC]

## What is it?

1. Region-Hierarchical is a baseline deep learning method for image paragraph captioning task.

2. Image paragraph captioning aims to generate a paragraph to describe the given image.

## Requirements

* python 3.7
* pytorch 1.3.1
* torchvision 0.4.2
* tensorboard 2.0.1
* h5py
* tqdm
* spacy (We use it to pre-process original data)
* Prefetch version of DataLoader: [https://github.com/IgorSusmelj/pytorch-styleguide/issues/5](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5)

## Data Processing

### Prepare Original Dataset

1. [Download Stanford Paragraph Dataset Here](https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html)

   * paragraph annotation file: `paragraphs_v1.json`
   * dataset splitting file:  `train_split.json` / `test_split.json` / `val_split.json`

2. Download images by `url` in `paragraphs_v1.json`. e.g.
   ```json
   {
   	"url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2388350.jpg",
       "image_id": 2388350,
       "paragraph": "On a white plastic cutting board is a bunch of chopped vegetables. On one side are chopped mushrooms they are a white color. On the other side are some bright green chopped broccoli. In the middle are bright orange chopped carrots."
   }
   ```
   
3. Create a new folder `./data/stanfordParagraph`  to store all unprocessed original data

   ```bash
   data
   └──stanfordParagraph
       ├── images
       └── paragraphs
   ```

### Extract Image Features

#### Do It Yourself

There are two widely adopted image encoders for paragraph captioning: [DenseCap](https://cs.stanford.edu/people/karpathy/densecap/) and [BottomUp](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf). They are responsible for detecting region of interests and encode them into dense vectors.

* For DenseCap features we refer you to [jcjohnson/densecap](https://github.com/jcjohnson/densecap)，for more detailed instructions please follow [Wentong-DST/im2p](https://github.com/Wentong-DST/im2p).
* For BottomUp features we refer you to [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)，also recommend to follow [airsplay/lxmert](https://github.com/airsplay/lxmert) who provides a docker solution. 

#### Download extracted ones

For convenience, we share the extracted features and its correspond bounding boxes at [OneDrive](https://1drv.ms/u/s!AmN4YCVEJTAIhdBolrVmKG24SJtSuw?e=nAd4Iv):

* DenseCap features are wrapped in `densecap.zip`
  * `im2p_xxx_output.h5` contains bounding box feature and coordinates.
  * `imgs_xxx_path.txt` are the mappings between images and `.h5` file.
* BottomUP features have two versions：
  * `bu.zip` : 10 to 100 features per image (adaptive)
  * `bu36.zip`：36 features per image (fixed)
  * We adopt the `scripts/make_bu_data.py` provided by [ruotianluo/ ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) to transform `tsv` into `npy`. Specifically, change the `infiles` at line 27 will work fine.

### Pre-process Image & Text Input

All pre-process functions are provided in `preprocess.py`. Note that the annotations in Stanford Paragraph  Dataset are not as clean as the ones in MSCOCO. Please follow the orders  below to generate processed files (See `preprocess.py` for more details):

1. Create a new folder `cleaned` under `./data`
2. Create preprocessed paragraph file
3. Create mappings
4. Create vocab file
5. Encode paragraphs
6. Map densecap features

Also for convenience, we share the generated files except step 6 at [OneDrive](https://1drv.ms/u/s!AmN4YCVEJTAIhdBt_QVHqMk28zvziQ?e=RepQ4M)

## Approach

### Model

[A Hierarchical Approach for Generating Descriptive Image Paragraphs](https://arxiv.org/abs/1611.06607)

![image](./pics/framework.png)

**The model contains three parts**: 

* A region detector (we have done this by pre-extracting features)
* An encoder to project and pooling regional features: `./model/encoder.py`.

* A hierarchical decoder with a Sentence-level RNN and a Word-level RNN: `./model/decoder.py`

**Difference between our implementation and the original paper**:

We observe gradient vanishing problem during training the hierarchical RNN, we introduced the following techniques to mitigate it:

* Use [Highway Network](https://arxiv.org/abs/1505.00387) to replace simple `MLP + ReLU`
* Adopt Merge strategy in Word-level RNN suggested in this [paper](https://arxiv.org/abs/1703.09137)

### Training

TBD

### Inference

TBD

### Evaluating

1. We utilize [Illuminati91/pycocoevalcap](https://github.com/Illuminati91/pycocoevalcap) to evaluate the model. It is a python 3.x version of the widely used [coco-eval](https://github.com/tylin/coco-caption). Corresponding code is organized in folder `./pycocoevalcap` and `./utils/coco.py`.

2. To obtain COCO format ground truth file, we adopt the script `./utils/prepro_captions.py` from [lukemelas/image-paragraph-captioning](https://github.com/lukemelas/image-paragraph-captioning). And put files into folder `./data/coco_gt` 

3. To evaluate a trained paragraph model

   ```
   python evaluate_coco.py --model_name debug --model_check_point debug.pth.
   ```

   

## Performance

TBD

