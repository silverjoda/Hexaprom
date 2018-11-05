import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, act_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def model_to_w(model):

    pass

def w_to_model(w, model):
    pass

def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

def _gather_flat_grad(self):
    views = []
    for p in self._params:
        if p.grad is None:
            view = p.data.new(p.data.numel()).zero_()
        elif p.grad.data.is_sparse:
            view = p.grad.data.to_dense().view(-1)
        else:
            view = p.grad.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def _add_grad(self, step_size, update):
    offset = 0
    for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
        offset += numel
    assert offset == self._numel()

model = Model(10, 6)
initial_params = model.parameters()
sd = model.state_dict()
