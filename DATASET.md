# Dataset Preparation
The total images of unseen classes will take about 1.1T. The extracted feature will also be over 100G. It is recommended to save them to your large space storage $IMAGE_FOLDER $FEAT_FOLDER and soft link them.
```Shell
cd ..
ln -s $IMAGE_FOLDER images
ln -s $FEAT_FOLDER feats
cd src
```

## Download Images of Unseen Class from ImageNet.

To get access to original images from ImageNet, you need to [register](http://image-net.org/signup) a pair of username and access key.
```Shell
python tools/download_image.py --user $YOUR_USER_NAME --key $YOUR_ACCESS_KEY --save_dir ../images/ --hop 2 &
```

**Note**:
- Downloading images of all unseen classes will take up to one day. We only download the 2-hops subset by default. You may choose to download 3-hops subset of all unseen classes by settting --hop to 3 or all.
- The script supports multi-process. It is recommended to run several copies.
- `img-3-hops.txt`, `img-all.txt` will be made after running `download_image.py --hop 3 / all`. Still, we provide the image list [here](https://drive.google.com/open?id=1br9dS99LeiNJB0S2NvzJLHl1xWpX5jQS)

## Extract Visual Features of Images
1. **Download** ImageNet preatrained CNN (resnet 50)
```Shell
mkdir ../pretrain_weights && cd ../pretrain_weights
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar xf resnet_v1_50_2016_08_28.tar.gz
cd ../src
```
To download  Inception pretrained model:
```Shell
cd ../pretrain_weights
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
tar xf inception_v1_2016_08_28.tar.gz
cd ../src
```
2. **Extract feature**. The following will give you default mode: extracting the feature with pretrained `res50` for `2-hops` and save them to `../feats/`.
The script supports multi-processing.  It is *recommended* to run several copies to speed up this step.
```Shell
python tools/extract_pool5.py --gpu $GPU_ID &
```
For full argument configurations:
```Shell
python tools/extract_pool5.py --fc $res50_OR_inception --model_path $CNN_MODEL_PATH --image_file $IMAGE_LIST_PATH --image_dir $DIR_TO_IMAGE --feat_dir $DIR_TO_SAVE_FEAT --gpu $GPU_ID &
```


## Build Graph and Prepare GCN Training Data
You can skip this step if you just want to use the GCN model we trained.
- **Build graph and convert to GCN input data.** The `tools/convert_to_gcn_data.py` converts semantic embedding to input $X$; prepare class relation graph; and convert classifier of pretrained CNN model to output $W$.
The output will be saved to ../data/$wv_$fc/
```Shell
python convert_to_gcn_data.py --fc res50 --wv glove
```
- We have provided most of the data lists we need. Though no need to run, the `tools/prepare_list.py`  demonstrates how to build those list file in `data/list/` from scratch.


## Extract Semantic Embedding
You can skip this step if you just want to use the GCN model we trained.
- **Downloading ours**.  The following command is for downloading *Glove* semantic embedding of ImageNet graph. The semantic embedding using the other two method can be downloaded from  [here](https://www.dropbox.com/sh/9pklcwm7rkhd9qa/AACDMMKHIMXNW5cmInFFrCDCa?dl=0).
``` Shell
mkdir ../data/word_embedding_model
wget -O ../data/word_embedding_model https://www.dropbox.com/s/b0f1le1hbs8p2b7/glove_word2vec_wordnet.pkl?dl=0
```
- **Extract yourself.** The script `tools/obtain_word_embedding.py` extracts 3 version of semantic embeddings. Please notice that:
    + GoogleNews and Fasttext embedding may require some other packages.
    + You need to download the pretrained word embedding dictionary. The link is written in the beginning of the script. Please manually download what you need since some model in the google drive cannot wget.
