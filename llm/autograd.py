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
