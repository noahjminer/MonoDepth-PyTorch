import matplotlib as mpl
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import PIL as pil

from PIL import Image
from main_monodepth_pytorch import Model
from easydict import EasyDict as edict
from utils import generate_disp_image, disp_to_depth
from torchvision import transforms

from main_monodepth_pytorch import post_process_disparity

dict_parameters_test = edict({'data_dir':'data',
                              'model_path':'model/monodepth_resnet18_001.pth',
                              'output_directory':'results/',
                              'input_height':768,
                              'input_width':1024,
                              'model':'resnet18_md',
                              'pretrained':False,
                              'mode':'predict',
                              'device':'cuda:0',
                              'input_channels':3,
                              'num_workers':4,
                              'use_multiple_gpu':False})

og_img = Image.open('data/camera_8_trim_Moment.jpg')
og_width = og_img.width
og_height = og_img.height

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with torch.no_grad():
    model = Model(dict_parameters_test)
    input_img = og_img.resize((1024, 768))

    disps = model.predict(input_img)
    disp = disps[0]

    # move input_img to tensor
    t_transform = transforms.ToTensor()
    input_img = t_transform(input_img)
    input_img = input_img[None, :, :, :]

    disp = disps[0][:, 0, :, :].unsqueeze(1)
    disparities = disp.squeeze().cpu().numpy()

    output = generate_disp_image(input_img.to(device), disp, device)
    output = np.transpose(output[0, :, :, :].cpu().detach().numpy(), (1, 2, 0))
    output = (output[:, :, :3] * 255).astype(np.uint8)
    
    img_transform = transforms.ToPILImage()
    output = img_transform(output)

    output.save('results/result.jpeg')

    disp_resized = torch.nn.functional.interpolate(disp, (og_height, og_width), mode="bilinear", align_corners=False)

    scaled_disp, depth = disp_to_depth(disp, 100, 1000)
    metric_depth = 5.4 * depth.cpu().numpy()

    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 75)

    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='Pastel2')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    print(colormapped_im.max())
    print(colormapped_im.min())
    im = Image.fromarray(colormapped_im)
    im.save('results/result2.jpeg')