import json
import os
path = '/data/dataset/UAV/coco_uav/annotations/'
test = json.load(open(f'{path}/test2017_sample.json'))
daylight, night, fog = dict(), dict(), dict()
daylight['categories'] = test['categories']
daylight['images'] = []
daylight['annotations'] = []

night['categories'] = test['categories']
night['images'] = []
night['annotations'] = []

fog['categories'] = test['categories']
fog['images'] = []
fog['annotations'] = []

attr_path = '/data/dataset/UAV/M_attr/test/'
att_id_value = dict()
for at in os.listdir(attr_path):
    f = open(attr_path+at)
    for line in f:
        info = line[:-1].split(',')[:3]
        att_id_value[at.split('_')[0]] = [int(_) for _ in info]
# 选择
d_id = []
n_id = []
f_id = []
for image in test['images']:
    file_name = image['file_name'].split('_')[0]
    if att_id_value[file_name][0]:
        d_id.append(image['id'])
        daylight['images'].append(image)
    if att_id_value[file_name][1]:
        n_id.append(image['id'])
        night['images'].append(image)
    if att_id_value[file_name][2]:
        f_id.append(image['id'])
        fog['images'].append(image)
print(len(d_id), len(n_id), len(f_id), len(test['images']))
for ann in test['annotations']:
    if ann['image_id'] in d_id:
        daylight['annotations'].append(ann)
    if ann['image_id'] in n_id:
        night['annotations'].append(ann)
    if ann['image_id'] in f_id:
        fog['annotations'].append(ann)
with open(f'{path}/test2017_sample_daylight.json', 'w') as f:
    json.dump(daylight, f, indent='\t')
with open(f'{path}/test2017_sample_night.json', 'w') as f:
    json.dump(night, f, indent='\t')
with open(f'{path}/test2017_sample_fog.json', 'w') as f:
    json.dump(fog, f, indent='\t')

    

