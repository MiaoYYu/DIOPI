# Copyright (c) 2023, DeepLink.

from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-2,
        rtol=1e-3,
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-3,
        rtol=1e-4,
    ),

    'baddbmm': dict(
        name=["baddbmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=["baddbmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },

            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'hardswish': dict(
        name=["hardswish"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d': dict(
        name=["adaptive_max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'pointwise_op': dict(
        name=['floor'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'erfinv': dict(
        name=["erfinv"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'silu': dict(
        name=["silu"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    ## when dtype of input is uint8, output might overflow.
                    "dtype": [Skip(Dtype.uint8)],
                },

            ],
        ),
    ),

    'div': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'div_rounding_mode': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'div_dtype_int_and_bool': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add', 'sub'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        atol=1e-1,
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        atol=1e-3,
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },

            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op_3': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'reduce_partial_op_zeros_input': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'reduce_partial_op_ones_input': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'select': dict(
        name=["select"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'masked_scatter': dict(
        name=["masked_scatter"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        para=dict(
            # The diopiEmbeddingRenorm_ function is temporarily unavailable due to the unsupported Cambrian operator.  
            # Thus, to pass the test case, skip all non-None types of the max_norm parameter in the configuration file.
            max_norm=[Skip(1.0)],   
        ),
    ),

    'clip_grad_norm': dict(
        name=["clip_grad_norm_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["grads"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'split': dict(
        name=["split"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'split_bool': dict(
        name=["split"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'sigmoid_focal_loss': dict(
        name=["sigmoid_focal_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['inputs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'nms': dict(
        name=["nms"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['scores'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'roi_align': dict(
        name=["roi_align"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'index': dict(
        name=["index"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'sgd_without_buf': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'adam': dict(
        name=['adam', 'adamw'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'unfold': dict(
        name=["unfold"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'unfold_int': dict(
        name=["unfold"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'cdist_compute_mode': dict(
        name=['cdist'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
    ),

    'argmax_same_value': dict(
        name=['argmax'],
    ),

    'adadelta': dict(
        name=["adadelta"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'rmsprop': dict(
        name=["rmsprop"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'smooth_l1_loss': dict(
        name=["smooth_l1_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'conv3d': dict(
        name=['conv3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'max_pool3d': dict(
        name=['max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d': dict(
        name=["adaptive_max_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'masked_select': dict(
        name=['masked_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'masked_select_not_float': dict(
        name=['masked_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'imum': dict(
        name=['maximum', 'minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'imum_broadcast': dict(
        name=['maximum', 'minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'imum_ones': dict(
        name=['maximum', 'minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'mm': dict(
        name=['mm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'index_fill': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'index_fill_scalar': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'pad': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'pad_not_float': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'constant_pad': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],

                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'constant_pad_positive': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'norm': dict(
        name=['norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'unique': dict(
        name=['unique'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'prod': dict(
        name=['prod'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'ctc_loss': dict(
        name=["ctc_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'remainder': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'remainder_tensor': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'remainder_other_zero': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'remainder_scalar': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'scatter': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'scatter_scalar': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'index_put_acc': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'index_put_acc_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'index_put': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        ),
    ),

    'index_put_acc_bool_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        ),
    ),

    'index_put_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'uniform': dict(
        name=['uniform'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.int64), Skip(Dtype.int32),
                              Skip(Dtype.int16), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        atol=1e-4,
    ),

    'copy': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        )
    ),

    'copy_different_dtype': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        )
    ),

    'interpolate': dict(
        name=["interpolate"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        )
    ),

    'col2im': dict(
        name=["col2im"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'im2col': dict(
        name=["im2col"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'cholesky': dict(
        name=['cholesky_ex'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
        requires_backward=[0],
        saved_args=dict(output=0),
    ),

    'triangular_solve': dict(
        name=['triangular_solve'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
        saved_args=dict(output=0),
    ),

    'normal': dict(
        name=["normal"],
        para=dict(
            mean=[Skip(0), Skip(0.1)],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'normal_std_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'normal_mean_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'normal_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'polar': dict(
        name=["polar"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['abs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
                {
                    "ins": ['angle'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        ),
    ),

}
