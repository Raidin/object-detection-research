import numpy as np
import os
import pandas as pd

class Dataset():
    def __init__(self):
        self.annotations = dict()
        self.images = dict()

    def AddImage(self, IMG_DIR):
        img_list = os.listdir(IMG_DIR)
        for i, img in enumerate(img_list):
            self.images[i] = img

    def AddAnnotation(self, key, value):
        self.annotations[key] = value

    def ParseAnnotations(self, ANN_DIR):
        for key in self.images.keys():
            ann_file = '{}.csv'.format(self.images[key].split('.')[0])
            ann = pd.read_csv(os.path.join(ANN_DIR, ann_file))

            gt_bboxes = np.array([], dtype=np.int32).reshape(0, 4)

            for row in ann.iterrows():
                line = row[1][0].split(" ")
                x1 = int(line[0])
                y1 = int(line[1])
                x2 = int(line[2])
                y2 = int(line[3])
                gt_bboxes = np.vstack([gt_bboxes, [x1, y1, x2, y2]])

            self.AddAnnotation(self.images[key], gt_bboxes)
