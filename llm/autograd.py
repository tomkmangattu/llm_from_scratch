import math
import random
from typing import Any
from itertools import chain

class Value:
    """Wraps a scalar and remembers how it was computed, so gradients can flow backward through it."""

    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op  # just a label, useful for debugging/printing
        self._backward = lambda: None  # default: leaf node, nothing to propagate

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):

        def propagate_grad(a_node, b_node, out):
            a_node.grad += out.grad
            b_node.grad += out.grad

        if isinstance(other, Value):
            b_node = other
        else:
            b_node = Value(other)
        out = Value(self.data + b_node.data, (self, b_node), "+")
        out._backward = lambda: propagate_grad(self, b_node, out)
        return out

    def __mul__(self, other):

        def propagate_grad(a_node, b_node, out):
            a_node.grad += out.grad * b_node.data
            b_node.grad += out.grad * a_node.data

        if isinstance(other, Value):
            otherValue = other
        else:
            otherValue = Value(other)
        out = Value(self.data * otherValue.data, (self, otherValue), "*")
        out._backward = lambda: propagate_grad(self, otherValue, out)
        return out

    def __pow__(self, n):

        def propagate_grad(a_node, n, out):
            a_node.grad += out.grad * n * a_node.data ** (n-1)

        out = Value(self.data ** n, (self,), "**")
        out._backward = lambda : propagate_grad(self, n, out)
        return out

    def exp(self):

        def propagate_grad(a_node, out):
            a_node.grad += out.grad * out.data
        
        out = Value(math.exp(self.data), (self,), "exp")
        out._backward = lambda : propagate_grad(self, out)
        return out

    def tanh(self):

        def propagate_grad(a_node, out):
            a_node.grad += out.grad * (1 - out.data**2)

        out = Value(math.tanh(self.data), (self,), "tanh")
        out._backward = lambda : propagate_grad(self, out)
        return out

    def backward(self):
        topo: list[Value] = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                [build_topo(node) for node in v._prev]
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        [node._backward() for node in reversed(topo)]

class Neuron:

    def __init__(self, nin: int) -> None:
        self.w = [ Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        result = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return result.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin: int, nout: int) -> None:
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        if self.nout == 1:
            return self.neurons[0](x)
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return list(chain.from_iterable(neuron.parameters() for neuron in self.neurons))


class MLP:

    def __init__(self, nin: int, nouts: list[int]) -> None:
        self.layers : list[Layer] = []
        for nout in nouts:
            self.layers.append(Layer(nin, nout))
            nin = nout

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))
