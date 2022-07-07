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


def preprocess_input(x, dim_ordering='default'):
    '''
    imagenet preprocessing
    '''
    if dim_ordering == 'default':
        # dim_ordering = K.image_data_format()
        dim_ordering = "channels_last"
    assert dim_ordering in {'channels_last', 'channels_first'}

    if dim_ordering == 'channels_first':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def print_model_summary(model: nn.Module):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def inference(args):
    t1 = timer()
    # Load image and exemplar patch.
    im = Image.open(args.im).convert('RGB')
    vis_im = im.resize((im.size[0]//4, im.size[1]//4))
    im = np.array(im)
    vis_im = np.array(vis_im)
    patch = np.array(Image.open(args.exemplar).convert('RGB'))
    if patch.shape[0] != 63 or patch.shape[1] != 63:
        raise Exception('The exemplar patch should be size 63x63.')

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

    # set up data
    im_pre = preprocess_input(im[np.newaxis, ...].astype('float'))
    patch_pre = preprocess_input(patch[np.newaxis, ...].astype('float'))
    data = {'image': im_pre,
            'image_patch': patch_pre}
    vis_im = vis_im / 255.

    # inference
    with torch.inference_mode():
        x = (torch.tensor(data["image"].copy(), dtype=torch.float).permute(0, 3, 1, 2),
             torch.tensor(data["image_patch"].copy(), dtype=torch.float).permute(0, 3, 1, 2))
        logits = model.forward(x=x)["logits"]
        pred = logits[0, 0, :vis_im.shape[0], :vis_im.shape[1]].cpu().detach().numpy()
        # pred = model.predict(data)[0, :vis_im.shape[0], :vis_im.shape[1]]
    print('Count by summation: %0.2f' % (pred.sum()/100.))
    end_time = timer()
    print(f'Total time spent: {timedelta(seconds=end_time-t1)}, Prediction time spent: {timedelta(seconds=end_time-t2)}.')

    vis_im *= .5
    vis_im[..., 1] += pred[..., 0]/5.
    vis_im = np.clip(vis_im, 0, 1)
    vis_im = (vis_im*255).astype(np.uint8)
    vis_im = Image.fromarray(vis_im)
    outpath = 'heatmap_vis.jpg'
    vis_im.save(outpath)
    print('Predicted heatmap visualization saved to %s' % outpath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', default='images/cells.jpg', type=str, help='path to image')
    parser.add_argument('--exemplar', default='images/exemplar_cell.jpg', type=str, help='path to exemplar patch')
    args = parser.parse_args()

    inference(args)
