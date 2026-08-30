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
            # += because a node can feed into multiple outputs; gradients from each must accumulate, not overwrite.
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
            # local derivative of a*b w.r.t. each factor is the *other* factor's data.
            a_node.grad += out.grad * b_node.data
            b_node.grad += out.grad * a_node.data

        if isinstance(other, Value):
            otherValue = other
        else:
            otherValue = Value(other)
        out = Value(self.data * otherValue.data, (self, otherValue), "*")
        out._backward = lambda: propagate_grad(self, otherValue, out)
        return out

    def __sub__(self, other):
        # expressed via __add__/__mul__ so subtraction gets correct gradient propagation for free.
        if isinstance(other, Value):
            b_node = other
        else:
            b_node = Value(other)
        return self + (b_node * -1)

    def __pow__(self, n):
        # n is a plain number (int/float), not a Value, so only self needs a gradient here.
        def propagate_grad(a_node, n, out):
            a_node.grad += out.grad * n * a_node.data ** (n-1)

        out = Value(self.data ** n, (self,), "**")
        out._backward = lambda : propagate_grad(self, n, out)
        return out

    def exp(self):

        def propagate_grad(a_node, out):
            # d(e^x)/dx = e^x = out.data, so the derivative is expressed via the output, not the input.
            a_node.grad += out.grad * out.data
        
        out = Value(math.exp(self.data), (self,), "exp")
        out._backward = lambda : propagate_grad(self, out)
        return out

    def tanh(self):

        def propagate_grad(a_node, out):
            # d(tanh x)/dx = 1 - tanh(x)^2 = 1 - out.data**2, again defined in terms of the output.
            a_node.grad += out.grad * (1 - out.data**2)

        out = Value(math.tanh(self.data), (self,), "tanh")
        out._backward = lambda : propagate_grad(self, out)
        return out

    def backward(self):
        topo: list[Value] = []
        visited = set()

        def build_topo(v: Value):
            # post-order DFS: a node is appended only after all its children, so topo is a valid topological order.
            if v not in visited:
                visited.add(v)
                [build_topo(node) for node in v._prev]
                topo.append(v)

        build_topo(self)

        self.grad = 1.0  # seed: d(self)/d(self) = 1
        # reversed(topo): propagate from outputs back to inputs, so every node's grad is fully accumulated before it fires.
        [node._backward() for node in reversed(topo)]

class Neuron:

    def __init__(self, nin: int) -> None:
        self.w = [ Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # start=self.b folds the bias into the sum and avoids sum()'s default int(0) start,
        # which would crash since Value has no __radd__.
        result = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return result.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin: int, nout: int) -> None:
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # a single-neuron layer returns a bare Value instead of a length-1 list,
        # so a final output layer chains cleanly into loss math like `ypred - ygt`.
        if self.nout == 1:
            return self.neurons[0](x)
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        # flattens each neuron's own [w..., b] list into one single flat list of Values.
        return list(chain.from_iterable(neuron.parameters() for neuron in self.neurons))


class MLP:

    def __init__(self, nin: int, nouts: list[int]) -> None:
        self.layers : list[Layer] = []
        for nout in nouts:
            self.layers.append(Layer(nin, nout))
            nin = nout  # next layer's input size is this layer's output size

    def __call__(self, x) -> Any:
        # feed forward: each layer's output becomes the next layer's input.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # flattens every layer's params into one flat list, for zero-grad / optimizer update loops.
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))
