import os
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import ToTensor

sys.path.append('./src')
import numpy as np
from PIL import Image

from timeit import default_timer as timer
from datetime import timedelta
from model import Generic_Matching_Net, config
from data import collate_fn, CarPKDataset


def print_model_summary(model: nn.Module):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def heatmap_image(image: torch.Tensor, output_map: torch.Tensor):
    scaled_output_map = torch.nn.functional.interpolate(input=output_map.unsqueeze(0), size=image.shape[1:], mode="bilinear")
    norm_output_map = (scaled_output_map/scaled_output_map.max()).squeeze(0).squeeze(0)
    a = 0.3
    b = 1.0 - a
    # image[:, :, :] = 0
    image[0, :, :] = (a * image[0] + b * norm_output_map)
    heatmap_tensor = torch.clip(image * 255.0, min=0, max=255)
    heatmap = heatmap_tensor.permute((1, 2, 0)).numpy().astype(np.uint8)
    im = Image.fromarray(heatmap)
    return im


def inference(args):
    t1 = timer()
    data_root = "/media/cwei/WD_BLACK/datasets/CARPK/CARPK_devkit/data/"
    dataset = CarPKDataset(data_root=data_root, data_meta_dir="./datasets/meta/", mode="valid")
    example = dataset[10]
    true_count = example["output_map"].sum()
    image_path = example["search_img_path"]
    batch = collate_fn(batch=[example])
    # load trained model
    model = Generic_Matching_Net(config=config)
    pretrained_state_dict = torch.load("./car_adapt/checkpoints/model.0025.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=pretrained_state_dict)

    t2 = timer()
    # inference
    with torch.inference_mode():
        logits = model.forward(x=batch["images"])["logits"]
        pred = logits[0].cpu().detach()
    print(f"Pred count: {pred.sum()/100.:.2f}, True count: {true_count/100.:.2f}")
    end_time = timer()
    print(f'Total time spent: {timedelta(seconds=end_time-t1)}, Prediction time spent: {timedelta(seconds=end_time-t2)}.')

    print(Path(image_path))
    image = ToTensor()(Image.open(image_path))
    heatmap = heatmap_image(image=image, output_map=pred)
    outpath = 'heatmap_vis.jpg'
    heatmap.save(outpath)
    print('Predicted heatmap visualization saved to %s' % outpath)


if __name__ == '__main__':
    import argparse
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', default='images/cells.jpg', type=str, help='path to image')
    parser.add_argument('--exemplar', default='images/exemplar_cell.jpg', type=str, help='path to exemplar patch')
    args = parser.parse_args()

    inference(args)
