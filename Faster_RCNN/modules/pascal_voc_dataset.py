import json
import os
import xmltodict

from tqdm import tqdm

class Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.year = 2012
        self.imageset = 'trainval'

        self.data_dir = os.path.join(self.root_dir, 'data', 'PascalVOC/VOCdevkit/VOC{}'.format(self.year))

        self.ann_path = os.path.join(self.data_dir, 'Annotations')
        self.img_path = os.path.join(self.data_dir, 'JPEGImages')

        self.train_imgset = os.path.join(self.data_dir, 'ImageSets', 'Main', 'train.txt')
        self.trainval_imgset = os.path.join(self.data_dir, 'ImageSets', 'Main', 'trainval.txt')
        self.val_imgset = os.path.join(self.data_dir, 'ImageSets', 'Main', 'val.txt')
        # self.test_imgset = os.path.join(self.data_dir, 'ImageSets', 'Main', 'test.txt')
        self.img_files = []
        self.ann_datas = []
        self.classes_count = dict()
        self.class_mapping = dict()

    def AddImage(self, year='2012', imageset='trainval'):
        self.year = year
        self.imageset = imageset
        self.data_dir = os.path.join(self.root_dir, 'data', 'PascalVOC/VOCdevkit/VOC{}'.format(self.year))

        if self.imageset == 'trainval':
            path = self.trainval_imgset
        elif self.imageset == 'train':
            path = self.train_imgset
        elif self.imageset == 'val':
            path = self.val_imgset

        with open(path) as f:
            for line in f:
                self.img_files.append(line.strip() + '.jpg')

    def Prepare(self):
        # Sample...[0:30]
        files = tqdm(self.img_files[:30])

        for idx, file in enumerate(files):
            files.set_description("Processing :: {}".format(file))

            ann_file = os.path.join(self.ann_path, '{}.xml'.format(file.split('.')[0]))

            with open(ann_file) as f:
                doc = xmltodict.parse(f.read())

            element = doc['annotation']

            annotation_data = dict()
            annotation_data['image_id'] = idx
            annotation_data['filepath'] = os.path.join(self.img_path, element['filename'])
            annotation_data['width'] = element['size']['width']
            annotation_data['height'] = element['size']['height']
            annotation_data['bboxes'] = []

            objects = element['object'] if isinstance(element['object'], list) else [element['object']]

            for obj in objects:
                obj_info = dict()

                class_name = obj['name']
                if class_name not in self.classes_count:
                    self.classes_count[class_name] = 1
                else:
                    self.classes_count[class_name] += 1

                # class mapping 정보 추가
                if class_name not in self.class_mapping:
                    self.class_mapping[class_name] = len(self.class_mapping)

                obj_info['class'] = class_name
                obj_info['x1'] = int(round(float(obj['bndbox']['xmin'])))
                obj_info['y1'] = int(round(float(obj['bndbox']['ymin'])))
                obj_info['x2'] = int(round(float(obj['bndbox']['xmax'])))
                obj_info['y2'] = int(round(float(obj['bndbox']['ymax'])))
                annotation_data['bboxes'].append(obj_info)

            self.ann_datas.append(annotation_data)

    def SaveAnnotation(self):
        ann_json = json.dumps(self.ann_datas, indent=4)
        with open('./pascal_voc_{}_{}_data.json'.format(self.year, self.imageset), 'w') as f:
            f.write(ann_json)
