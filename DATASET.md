# Dataset Preparation
The total images will take about 1.1T. The extracted feature will also be over 30G. It is recommended to save them to your large space storage $IMAGE_FOLDER $FEAT_FOLDER and soft link them.
```Shell
cd ..
ln -s $IMAGE_FOLDER images
ln -s $FEAT_FOLDER feats
cd src
```

## Download images of unseen class from ImageNet.

To get access to original images from ImageNet, you need to [register](http://image-net.org/signup) a pair of username and access key.
```Shell
python tools/download_image.py --user $YOUR_USER_NAME --key $YOUR_ACCESS_KEY --save_dir ../images/ --hop 2 &
```

**Note**:
- Downloading images of all unseen classes will take up to one day. We only download the 2-hops subset. You may choose to download 3-hops subset of all unseen classes by settting --hop to 3 or all.
- The script supports multi-process. It is recommended to run several copies.

## Extract visual features of images
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
