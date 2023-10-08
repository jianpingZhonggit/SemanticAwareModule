import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
# 检测结果
# plt.subplot(2, 3, 1)
# s = 100
# gt  = cv2.imread('gt.png')
# h, w, _ = gt.shape
# gt = cv2.resize(gt, (w+s,h+s))
# plt.imshow(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
# plt.axis('off')
# plt.subplot(2, 3, 4)
# pred  = cv2.imread('pred.png')
# h, w, _ = pred.shape
# pred = cv2.resize(pred, (w+s,h+s))
# plt.imshow(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
# plt.axis('off')
# 左上
# tl_gt = np.load('./tl_heat_gt.npy')
# print(tl_gt)
# # exit(0)
# flag = (tl_gt[:, :, 0] + tl_gt[:, :, 1] + tl_gt[:, :, 2]) <= 1.5
# tl_gt[flag] = 1
# tl_gt *= 255
# tl_gt = tl_gt.astype(np.uint8)
# print(tl_gt)
# # plt.subplot(2, 3, 2)
# plt.imshow(tl_gt)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('tl_gt.png')
# exit(0)
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
tl_pred = np.load('./tl_heat_pred.npy')
# plt.grid(axis='y')
# plt.grid(axis='x')
# # print(tl_pred)
topk = 30
tmp = tl_pred.reshape(-1,)
tmp = sorted(tmp, reverse=True) 
flag = (tl_pred[:, :, 0] + tl_pred[:, :, 1] + tl_pred[:, :, 2]) < tmp[topk]
tl_pred[flag, :] = (1, 1, 1)
tl_pred *= 255
tl_pred = tl_pred.astype(np.uint8)
# cv2.imwrite('tmp.png', tl_pred)
# img = cv2.imread('tmp.png')

# plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(tl_pred, cv2.COLOR_RGB2BGR))
plt.xticks([])
plt.yticks([])
plt.savefig('tl_pred.png')
exit(0)
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.grid(axis='y')
# plt.grid(axis='x')
# 右下
# tl_gt = np.load('./br_heat_gt.npy')
# print(tl_gt)
# flag = (tl_gt[:, :, 0] + tl_gt[:, :, 1] + tl_gt[:, :, 2]) <= 1.5
# tl_gt[flag] = 1
# tl_gt *= 255
# tl_gt = tl_gt.astype(np.uint8)
# print(tl_gt)
# # plt.subplot(2, 3, 3)
# plt.imshow(tl_gt)
# plt.xticks([])
# plt.yticks([])
tl_pred = np.load('./br_heat_pred.npy')
# plt.savefig('br_gt.png')
# # plt.axis('off')
# exit(0)
# plt.xticks([])
# plt.yticks([])
# plt.grid(axis='y')
# plt.grid(axis='x')
# # print(tl_pred)
print(type(tl_pred), tl_pred.shape)
tmp = tl_pred.reshape(-1,)
tmp = sorted(tmp, reverse=True) 
flag = (tl_pred[:, :, 0] + tl_pred[:, :, 1] + tl_pred[:, :, 2]) < tmp[topk]
tl_pred[flag, :] = (1, 1, 1)
tl_pred *= 255
tl_pred = tl_pred.astype(np.uint8)
# cv2.imwrite('tmp.png', tl_pred)
# img = cv2.imread('tmp.png')

# plt.subplot(2, 3, 6)
plt.xticks([])
plt.yticks([])
plt.imshow(cv2.cvtColor(tl_pred, cv2.COLOR_RGB2BGR))
plt.savefig('br_pred.png')
exit(0)
# plt.axis('off')
plt.grid(axis='y')
plt.grid(axis='x')
plt.xticks([])
plt.yticks([])
# plt.tight_layout()
plt.subplots_adjust(left=0.112, bottom=0.260, right=0.945, top=0.845, wspace=0.005, hspace=0.005)
plt.savefig('res1.png')
plt.show()