import os
from pathlib import Path
from typing import Tuple, Union

import keras.layers.convolutional
import numpy as np
import torch
from torch import nn

from model import Generic_Matching_Net, config, L2_Normalization
from keras_gmn import two_stream_matching_networks
from collections import namedtuple
import l2_norm


def load_keras_model(path_to_pretrained_weights: str):
    Config = namedtuple('Config', 'patchdims imgdims outputdims')
    cg = Config(patchdims=(63, 63, 3), imgdims=(800, 800, 3), outputdims=(200, 200, 3))
    model = two_stream_matching_networks(cg, sync=False, adapt=False)
    model.load_weights(path_to_pretrained_weights)
    return model


def save_pytorch_model(model: nn.Module, path: str):
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(obj=model.state_dict(), f=path)


def print_pytorch_model_summary(model: nn.Module):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def print_keras_model_summary(model):
    # print(model.summary())
    print(model.layers[6].summary())


def get_keras_layers(model: nn.Module, class_or_tuple_of_classes: Union[type, Tuple]):
    interested_layers = []
    for layer in model.layers:
        layers = []
        if hasattr(layer, "layers"):
            layers.extend(layer.layers)
        else:
            layers.append(layer)

        for l in layers:
            if isinstance(l, class_or_tuple_of_classes):
                interested_layers.append(l)
    return interested_layers


def get_keras_linear_and_conv_layers(model):
    return get_keras_layers(model, class_or_tuple_of_classes=(keras.layers.convolutional.Conv2D, keras.layers.Dense))


def get_keras_batch_norm_layers(model):
    return get_keras_layers(model, class_or_tuple_of_classes=keras.layers.BatchNormalization)


def get_pytorch_layers(model: nn.Module, class_or_tuple_of_classes: Union[type, Tuple]):
    modules = list(model.modules())
    interested_layers = []
    for layer in modules:
        if isinstance(layer, class_or_tuple_of_classes):
            interested_layers.append(layer)
    return interested_layers


def get_pytorch_linear_and_conv_layers(model: nn.Module):
    return get_pytorch_layers(model, class_or_tuple_of_classes=(nn.Linear, nn.Conv2d, nn.ConvTranspose2d))


def get_pytorch_batch_norm_layers(model: nn.Module):
    return get_pytorch_layers(model, class_or_tuple_of_classes=nn.BatchNorm2d)


def sync_model1(from_keras_model, to_pytorch_model: nn.Module):
    keras_layers = get_keras_linear_and_conv_layers(from_keras_model)
    pytorch_layers = get_pytorch_linear_and_conv_layers(to_pytorch_model)
    assert len(keras_layers) == len(pytorch_layers)
    counter = 0
    for k_layer, p_layer in zip(keras_layers, pytorch_layers):
        # print(f"counter: {counter}")
        weight = torch.from_numpy(np.transpose(k_layer.get_weights()[0]))
        bias = torch.from_numpy(np.transpose(k_layer.get_weights()[1]))
        if weight.shape != p_layer.weight.data.shape:
            print(f"weight shape does not match at layer '{counter+1}'")
        assert weight.shape == p_layer.weight.data.shape
        assert bias.shape == p_layer.bias.data.shape
        p_layer.weight.data = weight
        p_layer.bias.data = bias
        counter += 1


def sync_model2(from_keras_model, to_pytorch_model: nn.Module):
    keras_layers = get_keras_batch_norm_layers(from_keras_model)
    pytorch_layers = get_pytorch_batch_norm_layers(to_pytorch_model)
    assert len(keras_layers) == len(pytorch_layers)
    counter = 0
    for k_layer, p_layer in zip(keras_layers, pytorch_layers):
        gamma = torch.from_numpy(np.transpose(k_layer.get_weights()[0]))
        beta = torch.from_numpy(np.transpose(k_layer.get_weights()[1]))
        moving_mean = torch.from_numpy(np.transpose(k_layer.get_weights()[2]))
        moving_variance = torch.from_numpy(np.transpose(k_layer.get_weights()[3]))
        if gamma.shape != p_layer.weight.data.shape:
            print(f"weight shape does not match at layer '{counter+1}'")
        p_layer.weight.data = gamma
        p_layer.bias.data = beta
        p_layer.running_mean.data = moving_mean
        p_layer.running_var.data = moving_variance

        counter += 1


def sync_model3(from_keras_model, to_pytorch_model: nn.Module):
    keras_layers = get_keras_layers(
        from_keras_model,
        class_or_tuple_of_classes=l2_norm.L2_Normalization_Layer)
    pytorch_layers = get_pytorch_layers(
        to_pytorch_model,
        class_or_tuple_of_classes=L2_Normalization)
    assert len(keras_layers) == len(pytorch_layers)
    counter = 0
    for k_layer, p_layer in zip(keras_layers, pytorch_layers):
        alpha = torch.from_numpy(np.transpose(k_layer.get_weights()[0]))
        if alpha.shape != p_layer.alpha.data.shape:
            print(f"alpha shape does not match at layer '{counter+1}'")
        p_layer.alpha.data = alpha

        counter += 1


def convert_keras2pytorch(
        path_to_keras_pretrained_weights: str,
        folder_to_save_pytorch_weights: str):
    keras_model = load_keras_model(path_to_pretrained_weights=path_to_keras_pretrained_weights)
    print_keras_model_summary(keras_model)
    pytorch_model = Generic_Matching_Net(config=config)

    sync_model1(from_keras_model=keras_model, to_pytorch_model=pytorch_model)
    sync_model2(from_keras_model=keras_model, to_pytorch_model=pytorch_model)
    sync_model3(from_keras_model=keras_model, to_pytorch_model=pytorch_model)

    # print_pytorch_model_summary(model=pytorch_model)
    save_pytorch_model(model=pytorch_model, path=folder_to_save_pytorch_weights)


def set_working_dir():
    if os.getcwd().endswith("keras2pytorch"):
        parent_dir = Path(os.getcwd()).parent
        os.chdir(parent_dir)
    return os.getcwd()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    set_working_dir()
    convert_keras2pytorch(
        path_to_keras_pretrained_weights="./keras2pytorch/pretrained_gmn.h5",
        folder_to_save_pytorch_weights="./keras2pytorch/pretrained_gmn.pt"
    )
