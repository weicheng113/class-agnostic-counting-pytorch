import sys

import torch
from torch import nn

sys.path.append('./src')
import numpy as np
from PIL import Image

from timeit import default_timer as timer
from datetime import timedelta
from model import Generic_Matching_Net, config
from data import generic_tfms

def print_model_summary(model: nn.Module):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def inference(args):
    t1 = timer()
    # Load image and exemplar patch.
    im = Image.open(args.im).convert("RGB")
    patch = Image.open(args.exemplar).convert("RGB")
    if patch.size[0] != 63 or patch.size[1] != 63:
        raise Exception('The exemplar patch should be size 63x63.')

    im_tensor = generic_tfms(im)
    patch_tensor = generic_tfms(patch)
    # load trained model
    model = Generic_Matching_Net(config=config)
    # print("Model's state_dict:")
    # print_model_summary(model)
    # model_state_dict = model.state_dict()
    # print("Pretrained Model's state_dict:")
    pretrained_state_dict = torch.load("./keras2pytorch/pretrained_gmn.pt")
    # print_model_summary(model)
    model.load_state_dict(state_dict=pretrained_state_dict)

    t2 = timer()
    # inference
    with torch.inference_mode():
        x = (im_tensor, patch_tensor)
        logits = model.forward(x=x)["logits"]
        pred = logits[0].cpu().detach().numpy()
        # pred = model.predict(data)[0, :vis_im.shape[0], :vis_im.shape[1]]
    print('Count by summation: %0.2f' % (pred.sum()/100.))
    end_time = timer()
    print(f'Total time spent: {timedelta(seconds=end_time-t1)}, Prediction time spent: {timedelta(seconds=end_time-t2)}.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', default='images/cells.jpg', type=str, help='path to image')
    parser.add_argument('--exemplar', default='images/exemplar_cell.jpg', type=str, help='path to exemplar patch')
    args = parser.parse_args()

    inference(args)
