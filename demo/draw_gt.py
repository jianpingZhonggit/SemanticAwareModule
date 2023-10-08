import json
import cv2
path = '/data/dataset/UAV/coco_uav/'
val = json.load(open(path+'annotations/val2017_sample.json', 'r'))
image_id = 0
for image in val['images']:
    if image['file_name'] == 'M1009_img000095.jpg':
        image_id = image['id']
img = cv2.imread(path+'uav_sample/M1009_img000095.jpg')
color = [(45, 4, 210),(0, 255,0 ), (255, 0, 0)]
cnt = 0
for ann in val['annotations']:
    if ann['image_id'] == image_id:
        cnt += 1
        [x, y, w, h] = [int(_) for _ in ann['bbox']]
        cv2.rectangle(img, (x,y),(x+w, y+h), color=color[ann['category_id']-1], thickness=2)
print(cnt)
# cv2.imshow('gt', img)
cv2.imwrite('gt.png', img)
# cv2.waitKey(0)
