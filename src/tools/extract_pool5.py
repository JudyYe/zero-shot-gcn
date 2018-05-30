import argparse
import os
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import inception_v1
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1_base
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope
import time


def extract_feature(image_list, pool5, image_holder, preprocess, model_path, image_dir, feat_dir):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    
    init(model_path, sess)
    print('Done Init! ')
    net_time, cnt = 0, 0
    for i, index in enumerate(image_list):
        feat_name = os.path.join(feat_dir, index.split('.')[0] + '.npz')
        image_name = os.path.join(image_dir, index)
        lockname = feat_name + '.lock'
        if os.path.exists(feat_name):
            continue
        if os.path.exists(lockname):
            continue
        try:
            os.makedirs(lockname)
        except:
            continue
        t = time.time()
        cnt += 1
        image = preprocess(image_name)
        feat = run_feat(sess, pool5, image_holder, image)
        if not os.path.exists(os.path.dirname(feat_name)):
            try:
                os.makedirs(os.path.dirname(feat_name))
                print('## Make Directory: %s' % feat_name)
            except:
                pass
        np.savez_compressed(feat_name, feat=feat)
        net_time += time.time() - t
        if i % 1000 == 0:
            print('extracting feature [%d / %d] %s (%f sec)' % (i, len(image_list), feat_name, net_time / cnt * 1000), feat.shape)
            net_time = 0
            cnt = 0
        cmd = 'rm -r %s' % lockname
        os.system(cmd)


def init(model_path, sess):
    def get_variables_in_checkpoint_file(file_name):
        reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
        # reader.get_tensor()
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map, reader

    var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
    my_var_list = tf.global_variables()
    sess.run(tf.variables_initializer(my_var_list, name='init'))
    variables_to_restore = []
    my_dict = {}
    for v in my_var_list:
        name = v.name.split(':')[0]
        my_dict[name] = 0
        if not var_keep_dic.has_key(name):
            print('He does not have', name)
        else:
            if v.shape != var_keep_dic[name]:
                print('Does not match shape: ', v.shape, var_keep_dic[name])
                continue
            variables_to_restore.append(v)
    for name in var_keep_dic:
        if not my_dict.has_key(name):
            print('I do not have ', name)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, model_path)
    print('Initialized')


def preprocess_res50(image_name):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]
    width = image.shape[1]
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]

    image = image.astype(np.float32)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    image = image[np.newaxis, :, :, :]
    return image


def preprocess_inception(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]
    width = image.shape[1]
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]
    save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/debug/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir + '1.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = image.astype(np.float32)
    image /= 255
    image = 2 * image - 1
    image = image[np.newaxis, :, :, :]
    return image


def run_feat(sess, pool5, image_holder, image):
    feat = sess.run(pool5, feed_dict={image_holder: image})
    feat = np.squeeze(feat)
    # exit()
    return feat



def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }
  with arg_scope(
      [slim.conv2d],
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc


def inception_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'trainable': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }
  with arg_scope(
      [slim.conv2d],
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc


def res50():
    image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'image')
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv, end_point = resnet_v1.resnet_v1_50(image, global_pool=True, is_training=False)
    return net_conv, image


def inception():
    image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'image')
    with slim.arg_scope(inception_arg_scope(is_training=False)):
        with variable_scope.variable_scope(
                'InceptionV1', 'InceptionV1', [image, 1000], reuse=None) as scope:
            with arg_scope(
                    [layers_lib.batch_norm, layers_lib.dropout], is_training=False):
                net, end_points = inception_v1_base(image, scope=scope)
                with variable_scope.variable_scope('Logits'):
                    net_conv = layers_lib.avg_pool2d(
                        net, [7, 7], stride=1, scope='MaxPool_0a_7x7')
    print(net_conv.shape)

    return net_conv, image

def parse_arg():
    parser = argparse.ArgumentParser(description='word embeddign type')
    parser.add_argument('--fc', type=str, default='res50',
                        help='word embedding type: [inception, res50]')
    parser.add_argument('--model_path', type=str, default='../pretrain_weights/resnet_v1_50.ckpt',
                        help='path to pretrained model')
    parser.add_argument('--image_file', type=str, default='../data/list/img-2-hops.txt',
                        help='list of image file')
    parser.add_argument('--image_dir', type=str, default='../images/',
                        help='directory to save features')
    parser.add_argument('--feat_dir', type=str, default='../feats/',
                        help='directory to save features')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


args = parse_arg()

if args.fc == 'inception':
    pool5, image_holder = inception()
    preprocess = preprocess_inception
elif args.fc == 'res50':
    pool5, image_holder = res50()
    preprocess = preprocess_res50
else:
    raise NotImplementedError
image_list, label_list = [], []
with open(args.image_file) as fp:
    for line in fp:
        index, l = line.split()
        image_list.append(index)
        label_list.append(int(l))

if __name__ == '__main__':
    extract_feature(image_list, pool5, image_holder, preprocess, args.model_path, args.image_dir, args.feat_dir)
