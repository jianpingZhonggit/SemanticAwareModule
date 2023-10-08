import json
path = '/data/dataset/UAV/uav/annotations/'
test = json.load(open(path+'test2017_sample.json'))
val = dict()
val['categories'] = test['categories']
val['images'] = []
val['annotations'] = []
image_id = []
for image in test['images']:
    if image['file_name'] == 'M1009_img000095.jpg':
        image_id.append(image['id'])
        break
    image_id.append(image['id'])
img_len = min(10, len(image_id) // 2)
image_id = image_id[-img_len:]
print(len(image_id))
for image in test['images']:
    if image['id'] in image_id:
        val['images'].append(image)
for ann in test['annotations']:
    if ann['image_id'] in image_id:
        val['annotations'].append(ann)
with open(path+'val2017_sample.json', 'w') as f:
    json.dump(val, f, indent='\t')
