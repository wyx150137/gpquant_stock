"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy import stats

__all__ = ['make_function']


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def _relu(x):
    return np.maximum(x, 0)


def _power2(x):
    return x ** 2


def _power3(x):
    return x ** 3


def _rank(x):
    """
    :param x:
    :return: 截面排名分位数
    """
    return stats.rankdata(x, axis=1) / x.shape[1]


def _norm(x):
    """
    :param x:
    :return: 截面标准化
    """
    return stats.zscore(x, axis=0)


def _reverse(x):
    """
    :param x:
    :return: 截面均值翻转
    """
    m = np.nanmean(x, axis=1)
    ret = np.abs(x - m)
    return ret


def _reverse_pro(x):
    """
    :param x:
    :return: 截面均值翻转
    """
    m = np.nanmean(x, axis=1)[:, np.newaxis]
    ret = np.where(x > m, 1, -1)
    return ret


def _shift(arr, num, fill_value):
    ret = np.empty_like(arr)
    if num > 0:
        ret[:num] = fill_value
        ret[num:] = arr[:-num]
    elif num < 0:
        ret[num:] = fill_value
        ret[:num] = arr[-num:]
    else:
        ret[:] = arr
    return ret


def _delay(x, num):
    return _shift(x, num, np.nan)


def _delta(x):
    num = 1
    return x - _shift(x, num, np.nan)


def _rolling_window(a, window):
    shape = (a.shape[0] - window + 1, window, a.shape[-1])
    strides = (a.strides[0],) + a.strides
    a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling


def _ts_max(x, period):
    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.nanmax(x_rolling, axis=1)
    return ret


def _ts_min(x, period):
    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.nanmin(x_rolling, axis=1)
    return ret


def _ts_mean(x, period):
    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.nanmean(x_rolling, axis=1)
    return ret


def _ts_std(x, period):
    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.nanstd(x_rolling, axis=1)
    return ret


def _ts_skew(x, period):
    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = stats.skew(x_rolling, axis=1)
    return ret


def _ts_corr_20(x, y):
    """
    :param x:
    :param y:
    :return: 时间序列相关系数
    """

    w = 20

    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, w)
    y_rolling = _rolling_window(y, w)

    x_rolling_mean = np.nanmean(x_rolling, axis=1)[:, np.newaxis, :]
    y_rolling_mean = np.nanmean(y_rolling, axis=1)[:, np.newaxis, :]

    x_rolling_demean = x_rolling - x_rolling_mean
    y_rolling_demean = y_rolling - y_rolling_mean

    corr = (np.sum(x_rolling_demean * y_rolling_demean, axis=1) /
            np.sqrt(np.sum(x_rolling_demean ** 2, axis=1) * np.sum(y_rolling_demean ** 2, axis=1)))

    ret[w - 1:, :] = corr

    return ret


def _ts_rank(x, w):
    """
    :param x:
    :param w:
    :return: 时序排名
    """

    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, w)

    ts_rank = stats.rankdata(x_rolling, axis=1)
    ret[w - 1:, :] = ts_rank[:, -1, :]

    return ret


def _ts_weighted_ma(x, period):
    ret = np.full(x.shape, np.nan)

    weight = np.arange(1, period + 1, 1).astype(np.float64)
    weight /= np.sum(weight)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.average(x_rolling, axis=1, weights=weight)

    return ret


def _ts_mean_rank_20(x):
    period = 20
    ret = np.full(x.shape, np.nan)
    x = _rank(x)
    x_rolling = _rolling_window(x, period)
    ret[period - 1:, :] = np.nanmean(x_rolling, axis=1)

    return ret


def _if_else_then(x, y, value_a, value_b):
    return np.where(x < y, value_a, value_b)


def _sign(x):
    return np.sign(x)


def _ts_ms_20(x):
    w = 20
    ret1 = _ts_mean(x, w)
    ret2 = _ts_std(x, w)
    return ret1 / ret2


def _ts_max_min_ratio(x):
    w = 20
    ret1 = _ts_max(x, w)
    ret2 = _ts_min(x, w)
    return ret1 / ret2


def _ts_cov_20(x, y):
    """
    :param x:
    :param y:
    :return: x 和 y rolling 20 的协方差
    """

    w = 20

    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, w)
    y_rolling = _rolling_window(y, w)

    x_rolling_mean = np.nanmean(x_rolling, axis=1)[:, np.newaxis, :]
    y_rolling_mean = np.nanmean(y_rolling, axis=1)[:, np.newaxis, :]

    x_rolling_demean = x_rolling - x_rolling_mean
    y_rolling_demean = y_rolling - y_rolling_mean

    cov = (np.sum(x_rolling_demean * y_rolling_demean, axis=1) / w)

    ret[w - 1:, :] = cov
    return ret


def _ts_mean_between_diff_top_5_bot_5_20(x):
    """
    :param x:
    :return: 过去20个时序值中，最大的五个数字的均值减去最小的五个数字的均值
    """

    w = 20

    ret = np.full(x.shape, np.nan)
    x_rolling = _rolling_window(x, w)

    top_5 = np.partition(x_rolling, -5, axis=1)[:, -5:, :]
    bot_5 = np.partition(x_rolling, 5, axis=1)[:, :5, :]

    ret[w - 1:, :] = np.nanmean(top_5, axis=1) - np.nanmean(bot_5, axis=1)

    return ret


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)
relu1 = _Function(function=_relu, name='relu', arity=1)
sign1 = _Function(function=_sign, name='sign', arity=1)
power2_1 = _Function(function=_power2, name='power2', arity=1)
power3_1 = _Function(function=_power3, name='power3', arity=1)

rank1 = _Function(function=_rank, name='rank', arity=1)
reverse1 = _Function(function=_reverse, name='reverse', arity=1)
norm1 = _Function(function=_norm, name='norm', arity=1)
revers_pro1 = _Function(function=_reverse_pro, name='revers_pro', arity=1)
if_else_then4 = _Function(function=_if_else_then, name='if_else_then', arity=4)

delta1 = _Function(function=_delta, name='delta', arity=1)
ts_mean2 = _Function(function=_ts_mean, name='ts_mean', arity=2)
ts_std2 = _Function(function=_ts_std, name='ts_std', arity=2)
ts_min2 = _Function(function=_ts_min, name='ts_min', arity=2)
ts_max2 = _Function(function=_ts_max, name='ts_max', arity=2)
ts_delay2 = _Function(function=_delay, name='ts_delay', arity=2)
ts_skew2 = _Function(function=_ts_skew, name='ts_skew', arity=2)
ts_corr20_2 = _Function(function=_ts_corr_20, name='ts_corr_20', arity=2)
ts_cov_20_2 = _Function(function=_ts_cov_20, name='ts_cov_20', arity=2)
ts_rank2 = _Function(function=_ts_rank, name='ts_rank', arity=2)
ts_ms20_1 = _Function(function=_ts_ms_20, name='ts_ms_20', arity=1)
ts_weighted_ma2 = _Function(function=_ts_weighted_ma, name='ts_weighted_ma', arity=2)
ts_mean_rank20_1 = _Function(function=_ts_mean_rank_20, name='ts_mean_rank_20', arity=1)
ts_max_min_ratio_1 = _Function(function=_ts_max_min_ratio, name='ts_max_min_ratio', arity=1)
ts_mean_between_diff_top_5_bot_5_20 = _Function(function=_ts_mean_between_diff_top_5_bot_5_20,
                                                name='ts_mean_between_diff_top_5_bot_5_20', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'sig': sig1,
                 'relu': relu1,
                 'sign': sign1,
                 'power2': power2_1,
                 'power3': power3_1,

                 'rank': rank1,
                 'reverse': reverse1,
                 'norm': norm1,
                 'revers_pro': revers_pro1,
                 'if_else_then': if_else_then4,

                 'delta': delta1,
                 'ts_mean': ts_mean2,
                 'ts_std': ts_std2,
                 'ts_min': ts_min2,
                 'ts_max': ts_max2,
                 'ts_delay': ts_delay2,
                 'ts_skew': ts_skew2,
                 'ts_corr_20': ts_corr20_2,
                 'ts_rank': ts_rank2,
                 'ts_ms_20': ts_ms20_1,
                 'ts_weighted_ma': ts_weighted_ma2,
                 'ts_mean_rank_20': ts_mean_rank20_1,
                 'ts_cov_20': ts_cov_20_2,
                 }
