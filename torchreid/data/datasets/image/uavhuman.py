from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import re

from torchreid.data import ImageDataset


class UAVHuman(ImageDataset):
    """
    UAV-Human dataset
    """
    dataset_dir = 'uavhuman'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        super(UAVHuman, self).__init__(train, query, gallery, **kwargs)
    
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern_pid = re.compile(r'P([-\d]+)S([-\d]+)')
        pattern_camid = re.compile(r'A([-\d]+)R([-\d])_([-\d]+)_([-\d]+)')
        distractor_pid = 50000

        pid_container = set()
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                pid = int(distractor_pid)
            else:
                pid_part1, pid_part2 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2)
            
            if pid == -1: continue  # junk images are just ignored
            # if pid == 3109 or pid == 8405: 
            #     import ipdb; ipdb.set_trace()
            #     continue

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                pid = int(distractor_pid)
                camid = int(fname[-13:-8])
            else:
                pid_part1, pid_part2 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2)
                camid_part1, _, _, camid_part2 = pattern_camid.search(fname).groups()
                camid = int(camid_part1 + camid_part2)
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    