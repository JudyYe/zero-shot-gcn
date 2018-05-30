import argparse
import os
import threading
import urllib
import glob

data_dir = '../data/'


def download(start, end, todo_list):
    username = 'judyye'
    access_key = '0c1c85c8452cfc8f0857b12038b9025481a69d11'
    # username = 'YOUR_USER_NAME'
    # access_key = 'YOUR_ACCESS_KEY'
    url_list = 'http://www.image-net.org/download/synset?wnid='
    url_key = '&username=%s&accesskey=%s&release=latest&src=stanford' % (args.user, args.key)
    print url_key

    testfile = urllib.URLopener()
    num = 0
    for i in range(start, end):
        wnid = todo_list[i]
        url_acc = url_list + wnid + url_key
        cmd = '\"' + url_acc + '\"'
        if os.path.exists(os.path.join(scratch_dir, wnid)) or os.path.exists(os.path.join(scratch_dir, wnid) + '.tar'):
            print 'Exist!! %s ' % os.path.join(scratch_dir, wnid)
            num += 1
        else:
            print('%d, %s, Downloading ...' % (i, wnid))
            try:
                testfile.retrieve(url_acc, scratch_dir + '/' + wnid + '.tar')
            except:
                print('Error when downloading', wnid)
            if not os.path.exists(os.path.join(scratch_dir, wnid)):
                os.makedirs(os.path.join(scratch_dir, wnid))
            cmd = 'tar -xf ' + os.path.join(scratch_dir, wnid + '.tar') + ' --directory ' + os.path.join(scratch_dir,
                                                                                                         wnid)
            os.system(cmd)
            cmd = 'rm ' + os.path.join(scratch_dir, wnid + '.tar')
            os.system(cmd)


def multi_download(entity_file):
    todo_list = []
    with open(entity_file) as fp:
        for line in fp:
            todo_list.append(line[0:-1])
    num = args.num_threads
    interval = len(todo_list) / num
    thread_list = []
    for i in range(num):
        if i == num - 1:
            end = len(todo_list)
        else:
            end = (i + 1) * interval
        thread = threading.Thread(target=download, args=(i * interval, end, todo_list))
        thread.start()
        thread_list.append(thread)
    for i in range(num):
        thread_list[i].join()


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
    multi_download(list_file)

    make_image_list(list_file, args.save_dir, name)