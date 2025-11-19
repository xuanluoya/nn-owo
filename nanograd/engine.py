import math
from typing import Tuple, Union


class Value:
    # Value 类：用于构建一个计算图中的节点（神经网络中常见的标量节点）。
    # Value 类是自动微分系统（autograd engine）的基本节点，
    # 支持任意复杂的数学表达式的自动求导（通过构建计算图并反向传播梯度）。

    def __init__(
        self,
        data: float,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data = data  # 节点存储的实际数值
        self.grad = 0.0  # 当前节点的梯度（反向传播后得到）
        self._backward = (
            lambda: None
        )  # 会在每个有输入产生输出的小节点上执行链式法则，自动产生grad
        self._prev = set(_children)  # 该节点的直接前驱节点集合
        self._op = _op  # 生成该节点的操作符
        self.label = label  # 可选标签（如 'x', 'y', 'loss'）

    # 包装输出
    def __repr__(self):
        # 方便调试的输出：仅显示当前节点的数值。
        return f"Value(data={self.data})"

    # 基本算子重载
    # 加法
    def __add__(self, other):
        # 如果是一个普通的数字（1， 2， 3），我们将会把它包装成Value值。
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 为什么是+=？因为一个变量可能被调用两次。
            # 这样它的梯度就成了两条路径的梯度之和。
            # 直接使用=是覆盖梯度。
            # 在自动微分系统中，节点可能有多个父节点，因此必须使用累积梯度。
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # 处理减法
    def __sub__(self, other):
        return self + (-other)

    # 乘法
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # 除法
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 从数学上说，a/b = a * (1/b) = a * b**-1
        return self * other**-1

    # 幂运算
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers now."
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # 求导：dx/dy = n * x^(n-1)
            # x.grad = 上游梯度 * 导数
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # 调转运算
    def __radd__(self, other):
        return self + other

    # __remul__用来处理 2 * Value 的状态。
    # 当我们使用 Value * 2 的时候，解释器实际上看到的是 x.__mul__(2)
    # 这种情况下因为我们在代码中包装了它， 所以可以解决。
    # 但是当2.__mul__(x)时，我们无法解决2的类型问题。
    # 解释器会在这时候调用__rmul__，尝试交换两个数（将a*b变成b*a）
    # 这时候就又变成了x.__mul__(2)，问题解决。
    # 我们需要声明一下，告诉编译器我们知道怎么办：
    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        # other - self = other + (-self)
        return other + (-self)

    def __rtruediv__(self, other):
        # other / self = other * self**-1
        return self * other**-1

    # 处理负数数据（出现-x行为）
    def __neg__(self):
        return self * -1

    # 常用函数
    # 双曲正切函数
    def tanh(self):
        x = self.data
        # 自制的函数有溢出风险
        # t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    # 指数函数的前向计算和反向传播规则。
    def exp(self):
        x = self.data
        # 对应数学指数函数：y = e^x
        # 计算出 e^x 并存入节点集
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # x.grad = dx/dout * out.grad = e^x * out.grad
            # 我们的目标是计算出终极输出L对某个输入x的影响程度，
            # 也就是dx/dL，但是模型都是一小步一小步的，比如：L = ... = y = exp(x)
            # 所以要通过链式法则分解： dx/dL = dy/dL * dx/dy
            # dy/dL 是上游梯度，在正常状态下一被上游计算完，可以直接用，也就是out.gard。
            # dx/dy 是局部导数，也就是说我们要对 y = e^x 求导，
            # 根据维基：y = e^x => dx/dy = e^x，y求导后依旧是e^x
            # 所以最后就是 self.grad（自身梯度） = out.grad（上游梯度）* out.data（当前输出值e^x)
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    # 自动拓扑反向传播
    def backward(self):
        topo = []
        # 已访问的节点
        visited = set()

        def build_topo(v):
            # 递归地先访问节点的所有「前驱节点」 (_prev)，
            # 然后再把自己 (v) 加入到 topo 列表。
            if v not in visited:
                visited.add(v)
                for chiled in v._prev:
                    # 从左到右
                    # 递归调用自身检查每一个子节点
                    build_topo(chiled)
                    # [x1, w1, x2, w2, b, x1*w1, x2*w2, x1*w1 + x2*w2, n, o]
                    # 正向拓扑排序（topological order）
                topo.append(v)

        build_topo(self)

        # 输出对自己的梯度永远是 1
        self.grad = 1.0
        # 向前传播时数据从左往右流动（forward）
        # 反向传播时梯度传递方向就是从右往左流动（backward）
        # topo 列表是「正向拓扑顺序」。
        # 反向传播时，我们要从最终输出 o 开始，把梯度一层层传回去。
        # 所以要反转一下，不然就从x1开始backward，而x1后面没东西。
        for node in reversed(topo):
            node._backward()
