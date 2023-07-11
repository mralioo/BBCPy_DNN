import optuna
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from utils import np_to_var
from src.models.components.functions import identity, transpose_time_to_spat, squeeze_final_output
from src.models.components.modules import Expression, AvgPool2dWithConv, Ensure4d


class Deep4Net(nn.Sequential):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            in_chans,
            n_classes,
            input_window_samples,
            final_conv_length,
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=10,
            n_filters_3=100,
            filter_length_3=10,
            n_filters_4=200,
            filter_length_4=10,
            first_nonlin=elu,
            first_pool_mode="max",
            first_pool_nonlin=identity,
            later_nonlin=elu,
            later_pool_mode="max",
            later_pool_nonlin=identity,
            drop_prob=0.5,
            double_time_convs=False,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
            trial=None,
    ):
        if isinstance(trial, optuna.Trial):
            self.n_filters_time = trial.suggest_int("n_filters_time", low=25, high=75, step=15)
            self.filter_time_length = trial.suggest_int("filter_time_length", low=10, high=30, step=10)
            self.n_filters_spat = trial.suggest_int("n_filters_spat", low=25, high=75, step=15)
            self.pool_time_length = trial.suggest_int("pool_time_length", low=3, high=9, step=3)
            self.pool_time_stride = trial.suggest_int("pool_time_stride", low=2, high=6, step=2)
            self.n_filters_2 = trial.suggest_int("n_filters_2", low=25, high=75, step=15)
            self.filter_length_2 = trial.suggest_int("filter_length_2", low=10, high=30, step=10)
            self.n_filters_3 = trial.suggest_int("n_filters_3", low=80, high=120, step=20)
            self.filter_length_3 = trial.suggest_int("filter_length_3", low=10, high=30, step=10)
            self.n_filters_4 = trial.suggest_int("n_filters_4", low=180, high=220, step=20)
            self.filter_length_4 = trial.suggest_int("filter_length_4", low=10, high=30, step=10)
            # self.drop_prob = trial.suggest_float("drop_prob", 0,1)
        else:
            self.n_filters_time = n_filters_time
            self.filter_time_length = filter_time_length
            self.n_filters_spat = n_filters_spat
            self.pool_time_length = pool_time_length
            self.pool_time_stride = pool_time_stride
            self.n_filters_2 = n_filters_2
            self.filter_length_2 = filter_length_2
            self.n_filters_3 = n_filters_3
            self.filter_length_3 = filter_length_3
            self.n_filters_4 = n_filters_4
            self.filter_length_4 = filter_length_4
            # self.drop_prob = drop_prob

        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        self.add_module("ensuredims", Ensure4d())
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        self.add_module("conv_nonlin", Expression(self.first_nonlin))
        self.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        self.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
                model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            self.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                self.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            self.add_module("nonlin" + suffix, Expression(self.later_nonlin))

            self.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            self.add_module(
                "pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
            )

        add_conv_pool_block(
            self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_window_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        self.add_module("sigmoid", nn.Sigmoid())
        self.add_module("squeeze", Expression(squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch
    from torchsummary import summary

    # x = Variable(torch.from_numpy(np.random.randn(1, 44, 534)))
    # x = Variable(torch.from_numpy(np.random.randn(1, 62, 6000)))
    x = Variable(torch.zeros((64, 62, 6000)))
    in_chans = 62
    n_classes = 1
    input_window_samples = 6000
    final_conv_length = "auto"
    model = Deep4Net(in_chans,
                     n_classes,
                     input_window_samples,
                     final_conv_length,
                     n_filters_time=75,
                     n_filters_spat=75,
                     filter_time_length=20,
                     pool_time_length=9,
                     pool_time_stride=9,
                     n_filters_2=50,
                     filter_length_2=30,
                     n_filters_3=80,
                     filter_length_3=30,
                     n_filters_4=100,
                     filter_length_4=4,
                     first_nonlin=elu,
                     first_pool_mode="max",
                     first_pool_nonlin=identity,
                     later_nonlin=elu,
                     later_pool_mode="max",
                     later_pool_nonlin=identity,
                     drop_prob=0.5,
                     double_time_convs=False,
                     split_first_layer=True,
                     batch_norm=True,
                     batch_norm_alpha=0.1,
                     stride_before_pool=False, )

    summary(model, (62, 6000), device="cpu")
    # s = get_output_shape(model, 62, 6000)
    y_pred = model(x)
