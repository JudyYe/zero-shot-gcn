import json
import xml.etree.ElementTree as ET
import os
import pickle as pkl

hop_url = {'1k': 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt',
           '2-hops': 'https://norouzi.github.io/research/conse/1549-word-net-ids.txt',
           '3-hops': 'https://norouzi.github.io/research/conse/7860-word-net-ids.txt',
           'all': 'http://www.image-net.org/api/xml/ReleaseStatus.xml'}
word_url = {'words'    : 'http://image-net.org/archive/words.txt',
            'structure': 'http://www.image-net.org/api/xml/structure_released.xml'}

data_dir = '../../data/list/'


def download_list(data_dir):
    # words.txt: wnid - text
    file_path = os.path.join(data_dir, 'words.txt')
    dict_path = os.path.join(data_dir, 'words.pkl')
    if not os.path.exists(dict_path):
        cmd = 'wget -O %s %s' % (file_path, word_url['words'])
        os.system(cmd)
        print('Downloaded words.txt to %s' % file_path)
        wnid_word = {}
        with open(file_path) as fp:
            for line in fp:
                wn, name = line.split('\t')
                wnid_word[wn] = name.strip()
        with open(dict_path, 'w') as fp:
            pkl.dump(wnid_word, fp)
        print('Save wnid to text dictionary to %s' % dict_path)
    else:
        print('List existed: %s' % dict_path)

    # get structure of imagenet
    file_path = os.path.join(data_dir, 'structure_released.xml')
    if not os.path.exists(file_path):
        cmd = 'wget -O %s %s' % (file_path, word_url['structure'])
        os.system(cmd)
        print('Downloaded structure_released.xml to %s' % file_path)
    else:
        print('List existed: %s' % file_path)


def add_edge_dfs(node):
    edges = []
    vertice = [ node.attrib['wnid'] ]
    if len(node) == 0:
        return vertice, edges
    for child in node:
        if child.tag != 'synset':
            print child.tag

        edges.append((node.attrib['wnid'], child.attrib['wnid']))
        child_ver, child_edge = add_edge_dfs(child)
        edges.extend(child_edge)
        vertice.extend(child_ver)

    return vertice, edges


def find_neighbor(srcfile):
    tree = ET.parse(srcfile)
    root = tree.getroot()
    vertice, edge_list = add_edge_dfs(root[1])
    print 'num of edge: ', len(edge_list)
    print 'num of vert: ', len(vertice)
    vertice = list(set(vertice))
    print 'After deduplicate, num of vert: ', len(vertice)
    return vertice, edge_list


def convert_to_graph(vertice, edges):
    graph_file = os.path.join(data_dir, '../imagenet_graph.pkl')

    inv_wordn_file = os.path.join(data_dir, 'invdict_wordn.json')
    inv_wordn_word_file = os.path.join(data_dir, 'invdict_wordntext.json')

    # indict_wordn
    with open(inv_wordn_file, 'w') as fp:
        json.dump(vertice, fp)
        print('Save graph node in wnid to %s' % inv_wordn_file)

    ver_dict = {}
    graph = {}
    for i in range(len(vertice)):
        ver_dict[vertice[i]] = i
        graph[i] = []

    for i in range(len(edges)):
        if not ver_dict.has_key(edges[i][1]):
            print('no!!!', i)
        id1 = ver_dict[edges[i][0]]
        id2 = ver_dict[edges[i][1]]
        graph[id1].append(id2)
        graph[id2].append(id1)

    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save ImageNet structure to: ', graph_file)

    dict_path = os.path.join(data_dir, 'words.pkl')
    with open(dict_path) as fp:
        wnid_word = pkl.load(fp)

    words = []
    for i in range(len(vertice)):
        wnid = vertice[i]
        if wnid_word.has_key(wnid):
            words.append(wnid_word[wnid])
        else:
            words.append(wnid)
            print wnid

    with open(inv_wordn_word_file, 'w') as fp:
        json.dump(words, fp)
        print('Save graph node in text to %s' % inv_wordn_word_file)


def make_zero_shot_list(name):
    seen_file = os.path.join(data_dir, '1k.txt')
    if not os.path.exists(seen_file):
        cmd = 'wget -O %s %s' % (seen_file, hop_url['1k'])
        os.system(cmd)

    unseen_file = os.path.join(data_dir, name + '.txt')
    cmd = 'wget -O %s %s' % (unseen_file, hop_url[name])
    os.system(cmd)
    with open(unseen_file) as fp:
        test_list = [line.split()[1] for line in fp]
    with open(unseen_file, 'w') as fp:
        for line in test_list:
            fp.write('%s\n' % line)
    cmd = 'cat %s %s > %s' % (seen_file, unseen_file, os.path.join(data_dir, '1k-%s.txt' % name))
    os.system(cmd)


def make_zero_shot_20klist(name):
    seen_file = os.path.join(data_dir, '1k.txt')
    if not os.path.exists(seen_file):
        cmd = 'wget -O %s %s' % (seen_file, hop_url['1k'])
        os.system(cmd)

    seen_dict = {}
    with open(seen_file) as fp:
        for line in fp:
            seen_dict[line.strip()] = 0

    unseen_file = os.path.join(data_dir, '%s.txt' % name)
    status_file = os.path.join(data_dir, 'ReleaseStatus.xml')
    if not os.path.exists(status_file):
        cmd = 'wget -O %s %s' % (status_file, hop_url['all'])
        os.system(cmd)

    tree = ET.parse(status_file)
    syn_list = tree.findall('.//synset')
    print(len(syn_list))

    unseen_dict = {}
    unseen_list = []
    for syn in syn_list:
        wnid = syn.get('wnid')
        if not seen_dict.has_key(wnid) and not unseen_dict.has_key(wnid):
            unseen_dict[wnid] = 0
            unseen_list.append(wnid)
    print('all unseen: ', len(unseen_dict))

    with open(unseen_file, 'w') as fp:
        for wnid in unseen_list:
            fp.write('%s\n' % wnid)
    cmd = 'cat %s %s > %s' % (seen_file, unseen_file, os.path.join(data_dir, '1k-%s.txt' % name))
    os.system(cmd)


def make_corresp(name):
    inv_wordn_file = os.path.join(data_dir, 'invdict_wordn.json')
    with open(inv_wordn_file) as fp:
        json_data = json.load(fp)
    seen_file = os.path.join(data_dir, '1k.txt')
    unseen_file = os.path.join(data_dir, '%s.txt' % name)
    seen_dict = {}
    unseen_dict = {}
    with open(seen_file) as fp:
        cnt = 0
        for line in fp:
            seen_dict[line.strip()] = cnt
            cnt += 1
    with open(unseen_file) as fp:
        cnt = len(seen_dict)
        for line in fp:
            unseen_dict[line.strip()] = cnt
            cnt += 1

    corresp_list= []
    for i in range(len(json_data)):
        wnid = json_data[i]
        corresp_id = -1
        is_unseen = 0

        # this is 1k
        if seen_dict.has_key(wnid):
            corresp_id = seen_dict[wnid]

        # not in 1k, is it in unseen test list?
        if corresp_id == -1 and unseen_dict.has_key(wnid):
            corresp_id = unseen_dict[wnid]
            is_unseen = 1

        corresp_list.append([corresp_id, is_unseen])

    check_train, check_test = 0, 0
    # (-1, -) neither seen nor useen
    # (id, 0) seen
    # (id, 1) unseen
    for i in range(len(corresp_list)):
        if corresp_list[i][1] == 1:
            check_test += 1
            assert corresp_list[i][0] >= 1000
        elif corresp_list[i][0] > -1:
            check_train += 1
            assert corresp_list[i][1] == 0 and corresp_list[i][0] < 1000

    print('[unseen set %s] unseen: %d, seen: %d' % (name, check_test, check_train))
    save_file = os.path.join(data_dir, 'corresp-%s.json' % name)
    with open(save_file, 'w') as fp:
        json.dump(corresp_list, fp)
    return



def prepare_graph():
    struct_file = os.path.join(data_dir, 'structure_released.xml')
    vertice, edges = find_neighbor(struct_file)
    convert_to_graph(vertice, edges)


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print('## Make Directory: ', data_dir)
    download_list(data_dir)
    prepare_graph()

    make_zero_shot_list('2-hops')
    make_zero_shot_list('3-hops')
    make_zero_shot_20klist('all')

    make_corresp('2-hops')
    make_corresp('3-hops')
    make_corresp('all')