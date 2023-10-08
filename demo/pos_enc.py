import numpy as np
feat = 128
# 生成mask
mask = np.ones(shape=(30, ))
mask[25:] = 0
print(mask)
x_embed = np.cumsum(mask, 0)
t = np.arange(feat)
dim_t = 1000 ** (t / feat)
print(dim_t)
print(x_embed)
pos_x = x_embed[:, None] / dim_t
print(np.sin(pos_x[:, 0::2]).shape, '---')
print(np.cos(pos_x[:, 1::2]).shape)
pos = np.concatenate((np.sin(pos_x[:, 0::2]), np.cos(pos_x[:, 1::2])), axis=1)
print(pos_x)
print(pos_x.shape)
import seaborn as sns
import matplotlib.pyplot as plt
print(pos.shape)
heatmap = sns.heatmap(pos)
plt.show()
import torch.nn as nn
query_embedding = nn.Embedding(100, 128)
print(query_embedding.weight)