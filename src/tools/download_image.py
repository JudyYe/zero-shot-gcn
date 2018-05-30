import argparse
import os
import threading
import urllib
import glob

data_dir = '../data/'


def download(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    url_list = 'http://www.image-net.org/download/synset?wnid='
    url_key = '&username=%s&accesskey=%s&release=latest&src=stanford' % (args.user, args.key)

    testfile = urllib.URLopener()
    for i in range(len(vid_list)):
        wnid = vid_list[i]
        url_acc = url_list + wnid + url_key

        save_dir = os.path.join(scratch_dir, wnid)
        lockname = save_dir + '.lock'
        if os.path.exists(save_dir):
            continue
        if os.path.exists(lockname):
            continue
        try:
            os.makedirs(lockname)
        except:
            continue
        tar_file = os.path.join(scratch_dir, wnid + '.tar')
        try:
            testfile.retrieve(url_acc, tar_file)
            print('Downloading %s' % wnid)
        except:
            print('!!! Error when downloading', wnid)
            continue

        if not os.path.exists(os.path.join(scratch_dir, wnid)):
            os.makedirs(os.path.join(scratch_dir, wnid))
        cmd = 'tar -xf ' + tar_file + ' --directory ' + save_dir
        os.system(cmd)
        cmd = 'rm ' + os.path.join(tar_file)
        os.system(cmd)
        cmd = 'rm -r %s' % lockname
        os.system(cmd)

        if i % 10 == 0:
            print('%d / %d' % (i, len(vid_list)))


def make_image_list(list_file, image_dir, name, offset=1000):
    with open(list_file) as fp:
        wnid_list = [line.strip() for line in fp]

    save_file = os.path.join(data_dir, 'list', 'img-%s.txt' % name)
    wr_fp = open(save_file, 'w')
    for i, wnid in enumerate(wnid_list):
        img_list = glob.glob(os.path.join(image_dir, wnid, '*.JPEG'))
        for path in img_list:
            index = os.path.join(wnid, os.path.basename(path))
            l = i + offset
            wr_fp.write('%s %d\n' % (index, l))
        if len(img_list) == 0:
            print('Warning: does not have class %s. Do you forgot to download the picture??' % wnid)
    wr_fp.close()


def rm_empty(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    cnt = 0
    for i in range(len(vid_list)):
        save_dir = os.path.join(scratch_dir, vid_list[i])
        jpg_list = glob.glob(save_dir + '/*.JPEG')
        if len(jpg_list) < 10:
            print(vid_list[i])
            cmd = 'rm -r %s ' % save_dir
            os.system(cmd)
            cnt += 1
    print(cnt)


def parse_arg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hop', type=str, default='2',
                        help='choice of test difficulties: 2,3,all')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='path to save images')
    parser.add_argument('--user', type=str,
                        help='your username', required=True)
    parser.add_argument('--key', type=str,
                        help='your access key', required=True)
    parser.add_argument('--num_threads', type=int, default=32,
                        help='num of downloading threads')
    args = parser.parse_args()
    if args.save_dir is None:
        print('Please set directory to save images')
    return args


args = parse_arg()
scratch_dir = args.save_dir

if __name__ == '__main__':
    if args.hop == '2':
        name = '2-hops'
        list_file = os.path.join(data_dir, 'list/2-hops.txt')
    elif args.hop == '3':
        name = '3-hops'
        list_file = os.path.join(data_dir, 'list/3-hops.txt')
    elif args.hop == 'all':
        name = 'all'
        list_file = os.path.join(data_dir, 'list/all.txt')
    else:
        raise NotImplementedError
    download(list_file)

    make_image_list(list_file, args.save_dir, name)
