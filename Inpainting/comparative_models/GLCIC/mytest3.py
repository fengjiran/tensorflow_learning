import os
import argparse
import json
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import CompletionNetwork
from PIL import Image
from utils import poisson_blend, gen_input_mask

model_path = 'E:\\model\\comparative_models\\GLCIC\\celeba\\model_cn'
img_path = 'E:\\model\\experiments\\exp3\\celebahq\\gt_images'
# irregular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'
regular_mask_path = 'E:\\model\\experiments\\exp3\\celebahq\\mask\\128'

# irregular_output_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\GLCIC\\irregular'
regular_output_path = 'E:\\model\\experiments\\exp3\\celebahq\\results\\GLCIC\\128'


# ==============================================
# Load model
# ==============================================
with open('config.json', 'r') as f:
    config = json.load(f)
mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
model = CompletionNetwork()
if config['data_parallel']:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path, map_location='cuda'))


img_dir = os.listdir(img_path)
# irregular_mask_dir = os.listdir(irregular_mask_path)
regular_mask_dir = os.listdir(regular_mask_path)

# i = 1
# for dir1, dir2 in zip(img_dir, irregular_mask_dir):
#     img = Image.open(os.path.join(img_path, dir1))
#     x = transforms.ToTensor()(img)
#     x = torch.unsqueeze(x, dim=0)

#     mask = Image.open(os.path.join(irregular_mask_path, dir2))
#     mask = transforms.ToTensor()(mask)
#     mask = torch.unsqueeze(mask, dim=0)
#     mask = 1 - mask

#     filename = os.path.join(irregular_output_path, 'irregular_output_%04d.png' % i)

#     # inpaint
#     with torch.no_grad():
#         x_mask = x - x * mask + mpv * mask
#         inpt = torch.cat((x_mask, mask), dim=1)
#         output = model(inpt)
#         inpainted = poisson_blend(x, output, mask)
#         save_image(inpainted, filename)

#     i += 1
start = time.time()
i = 1
for dir1, dir2 in zip(img_dir, regular_mask_dir):
    img = Image.open(os.path.join(img_path, dir1))
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    mask = Image.open(os.path.join(regular_mask_path, dir2))
    mask = transforms.ToTensor()(mask)
    mask = torch.unsqueeze(mask, dim=0)
    mask = 1 - mask

    filename = os.path.join(regular_output_path, 'regular_output_%04d.png' % i)

    # inpaint
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        inpt = torch.cat((x_mask, mask), dim=1)
        output = model(inpt)
        inpainted = poisson_blend(x, output, mask)
        save_image(inpainted, filename)

    i += 1

end = time.time()
print((end - start) * 1000)
