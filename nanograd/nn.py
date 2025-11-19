import random

from nanograd.engine import Value


class Neuron:
    # nin 是神经元的输入数量，也就是能接受多少个输入，
    # 我们为每个输入构建一个权重，从-1到1之间的随机数，
    # 当然，还有必备的偏值。
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    # 当我们调用 f(x) 时，编译器会默认调用 __call__ 方法
    # 例如我们现在创建一个二维神经元和两个输入：
    # x = [1.0, 2.0]
    # n = Neuron(2)
    # n(x) -> 这时候就会调用 __call__ 方法。
    def __call__(self, x):
        # 我们首先要进行向前传播。
        # w * x + b
        # 将 w 的所有元素与 x 的所有元素两两相乘
        # zip将会创建一个迭代器，遍历输入进来的元组
        # 例如，w = [?, ?], x = [2, 3]，
        # zip将会创建一个list进行两两配对
        # [(Value(data=-0.1519860568157867), 2.0), (Value(data=-0.752612965018634), 3.0)]
        # 原始激活值：
        # 因为sum里有self.b所以实际上会从b的值开始累加
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # 进行非线性转换
        out = act.tanh()
        return out

    # 神经网络的所有参数
    def parameters(self):
        return self.w + [self.b]


# 神经元层，实际上就是一个神经元列表
class Layer:
    # nin代表有几个输入
    # nout代表你想拥有多少神经元在这一层（有几个输出）
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # 本层的所有参数
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params


# 现在我们开始制作多层感知器 MLP
class MLP:
    # 我们依旧接受输入数量，但不接受单个输出（也就是单层神经元的数量），
    # 我们现在会接受一个输出列表，这个列表定义了MLP中所有层的大小，
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
