import torch
import numpy as np

def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def generate_TS_channel_order(original_order: list):
    """https://github.com/yi-ding-cs/TSception/blob/main/code/utils.py

    This function will generate the channel order for TSception
    Parameters
    ----------
    original_order: list of the channel names

    Returns
    -------
    TS: list of channel names which is for TSception
    """
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        chan_name_len = len(channel)
        k = 0
        for s in [*channel[:]]:
            if s.isdigit():
               k += 1
        if k != 0:
            chan_name.append(channel[:chan_name_len-k])
            chan_num.append(int(channel[chan_name_len-k:]))
            chan_final.append(channel)
    chan_pair = []
    for ch, id in enumerate(chan_num):
        if id % 2 == 0:
            chan_pair.append(chan_name[ch] + str(id-1))
        else:
            chan_pair.append(chan_name[ch] + str(id+1))
    chan_no_duplicate = []
    [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if f not in chan_no_duplicate]
    return chan_no_duplicate[0::2] + chan_no_duplicate[1::2]
def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).

    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)


def get_output_shape(model, in_chans, input_window_samples):
    """Returns shape of neural network output for batch size equal 1.

    Returns
    -------
    output_shape: tuple
        shape of the network output for `batch_size==1` (1, ...)
    """
    with torch.no_grad():
        dummy_input = torch.ones(
            1, in_chans, input_window_samples,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        output_shape = model(dummy_input).shape
    return output_shape


