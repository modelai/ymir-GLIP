import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import os
def load(img_path):

    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(file,img, caption):
    plt.figure("Image")
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig('infer_result/'+'000700.jpg')
    plt.show()
    exit(0)

config_file = "configs/pretrain/glip_A_Swin_T_O365.yaml"
weight_file = "MODEL/glip_a_tiny_o365.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
    show_mask_heatmaps=False
)
glip_demo.color=(255,0,255)
for file in os.listdir('DATASET/VOC2028/val/'):
    print(file)
    if  os.path.splitext(file)[-1]!='.bmp':
        image = load('/in/assets/41/68624cc85d7515e9649d324d78bf875ed6dd9c41.jpg')
        caption = 'person'
        result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
        imshow(file,result, caption)
