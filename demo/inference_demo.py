
# coding: utf-8

# In[4]:


from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv


# In[5]:


# download the checkpoint demols
# get_ipython().system('mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest ./checkpoints')
config_file = 'work_dirs/reppoints-moment_r50_fpn_1x_coco/reppoints-moment_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/reppoints-moment_r50_fpn_1x_coco/epoch_1.pth'
# config_file = 'configs/centripetalnet/centripetalnet_hourglass52_16xb6-crop511-210e-mstest_uav.py'
# checkpoint_file = 'epoch_201_ours.pth'



# In[6]:


#Register all modules in mmdet into the registries
register_all_modules()
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'


# In[7]:


# test a single image
img_name = 'M1303_img000400'
img = mmcv.imread( f'demo/src/{img_name}.jpg', channel_order='rgb')
result = inference_detector(model, img)
print(result)


# In[8]:
print(model.cfg.visualizer)

# init the visualizer(execute this block only once)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta


# In[9]:


# show the results
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
    out_file=f'demo/result/{img_name}_cornernet.png'
)
visualizer.show()

