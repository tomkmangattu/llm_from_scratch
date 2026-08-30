import torch

from llm.autograd import Layer, MLP, Neuron, Value


def test_matches_pytorch_autograd():
    # Build L = (a * b + c) * f once with your Value, once with torch.
    a_mine, b_mine, c_mine, f_mine = Value(3.0), Value(4.0), Value(5.0), Value(2.0)
    L_mine = (a_mine * b_mine + c_mine) * f_mine
    L_mine.backward()

    a_torch = torch.tensor(3.0, requires_grad=True)
    b_torch = torch.tensor(4.0, requires_grad=True)
    c_torch = torch.tensor(5.0, requires_grad=True)
    f_torch = torch.tensor(2.0, requires_grad=True)

    L_torch = (a_torch * b_torch + c_torch) * f_torch
    L_torch.backward()

    assert(abs(L_mine.data - L_torch.item()) <1e-6 )

    assert a_torch.grad is not None
    assert(abs(a_mine.grad - a_torch.grad.item()) <1e-6)

    assert b_torch.grad is not None
    assert(abs(b_mine.grad - b_torch.grad.item()) <1e-6)

    assert c_torch.grad is not None
    assert(abs(c_mine.grad - c_torch.grad.item()) <1e-6)

    assert f_torch.grad is not None
    assert(abs(f_mine.grad - f_torch.grad.item()) <1e-6)


def test_pow_matches_pytorch_autograd():
    x_mine = Value(2.0)
    y_mine = x_mine ** 3
    y_mine.backward()

    x_torch = torch.tensor(2.0, requires_grad=True)
    y_torch = x_torch ** 3
    y_torch.backward()

    assert x_torch.grad is not None
    assert abs(y_mine.data - y_torch.item()) < 1e-6
    assert abs(x_mine.grad - x_torch.grad.item()) < 1e-6


def test_exp_matches_pytorch_autograd():
    x_mine = Value(0.5)
    y_mine = x_mine.exp()
    y_mine.backward()

    x_torch = torch.tensor(0.5, requires_grad=True)
    y_torch = x_torch.exp()
    y_torch.backward()

    assert x_torch.grad is not None
    assert abs(y_mine.data - y_torch.item()) < 1e-6
    assert abs(x_mine.grad - x_torch.grad.item()) < 1e-6


def test_tanh_matches_pytorch_autograd():
    x_mine = Value(0.5)
    y_mine = x_mine.tanh()
    y_mine.backward()

    x_torch = torch.tensor(0.5, requires_grad=True)
    y_torch = x_torch.tanh()
    y_torch.backward()

    assert x_torch.grad is not None
    assert abs(y_mine.data - y_torch.item()) < 1e-6
    assert abs(x_mine.grad - x_torch.grad.item()) < 1e-6


def test_neuron_matches_pytorch_autograd():
    n = Neuron(3)
    n.w[0].data, n.w[1].data, n.w[2].data = 0.5, -0.3, 0.8
    n.b.data = 0.1
    x = [2.0, 3.0, -1.0]

    out_mine = n(x)
    out_mine.backward()

    w_torch = torch.tensor([0.5, -0.3, 0.8], requires_grad=True)
    b_torch = torch.tensor(0.1, requires_grad=True)
    x_torch = torch.tensor(x)
    out_torch = torch.tanh((w_torch * x_torch).sum() + b_torch)
    out_torch.backward()

    assert abs(out_mine.data - out_torch.item()) < 1e-6

    assert w_torch.grad is not None
    for wi, wi_grad in zip(n.w, w_torch.grad):
        assert abs(wi.grad - wi_grad.item()) < 1e-6

    assert b_torch.grad is not None
    assert abs(n.b.grad - b_torch.grad.item()) < 1e-6


def test_layer_output_shape():
    # a layer with more than one neuron returns a list of outputs...
    layer_multi = Layer(3, 4)
    out_multi = layer_multi([1.0, 2.0, 3.0])
    assert isinstance(out_multi, list)
    assert len(out_multi) == 4
    assert all(isinstance(o, Value) for o in out_multi)

    # ...but a single-neuron layer returns a bare Value, not a length-1 list.
    layer_single = Layer(3, 1)
    out_single = layer_single([1.0, 2.0, 3.0])
    assert isinstance(out_single, Value)


def test_layer_parameters_are_flat():
    layer = Layer(3, 4)
    params = layer.parameters()
    assert len(params) == 4 * (3 + 1)
    assert all(isinstance(p, Value) for p in params)


def test_mlp_parameters_are_flat():
    mlp = MLP(3, [4, 4, 1])
    params = mlp.parameters()
    assert len(params) == 4 * (3 + 1) + 4 * (4 + 1) + 1 * (4 + 1)
    assert all(isinstance(p, Value) for p in params)


def test_mlp_matches_pytorch_autograd():
    mlp = MLP(2, [2, 1])

    mlp.layers[0].neurons[0].w[0].data = 0.1
    mlp.layers[0].neurons[0].w[1].data = -0.2
    mlp.layers[0].neurons[0].b.data = 0.05
    mlp.layers[0].neurons[1].w[0].data = 0.3
    mlp.layers[0].neurons[1].w[1].data = 0.4
    mlp.layers[0].neurons[1].b.data = -0.1
    mlp.layers[1].neurons[0].w[0].data = 0.2
    mlp.layers[1].neurons[0].w[1].data = -0.3
    mlp.layers[1].neurons[0].b.data = 0.15

    x = [1.0, -1.0]
    out_mine = mlp(x)
    out_mine.backward()

    w00_t = torch.tensor(0.1, requires_grad=True)
    w01_t = torch.tensor(-0.2, requires_grad=True)
    b0_t = torch.tensor(0.05, requires_grad=True)
    w10_t = torch.tensor(0.3, requires_grad=True)
    w11_t = torch.tensor(0.4, requires_grad=True)
    b1_t = torch.tensor(-0.1, requires_grad=True)
    w20_t = torch.tensor(0.2, requires_grad=True)
    w21_t = torch.tensor(-0.3, requires_grad=True)
    b2_t = torch.tensor(0.15, requires_grad=True)

    x0, x1 = x
    h0_t = torch.tanh(w00_t * x0 + w01_t * x1 + b0_t)
    h1_t = torch.tanh(w10_t * x0 + w11_t * x1 + b1_t)
    out_t = torch.tanh(w20_t * h0_t + w21_t * h1_t + b2_t)
    out_t.backward()

    assert abs(out_mine.data - out_t.item()) < 1e-6

    mine_params = [
        mlp.layers[0].neurons[0].w[0], mlp.layers[0].neurons[0].w[1], mlp.layers[0].neurons[0].b,
        mlp.layers[0].neurons[1].w[0], mlp.layers[0].neurons[1].w[1], mlp.layers[0].neurons[1].b,
        mlp.layers[1].neurons[0].w[0], mlp.layers[1].neurons[0].w[1], mlp.layers[1].neurons[0].b,
    ]
    torch_params = [w00_t, w01_t, b0_t, w10_t, w11_t, b1_t, w20_t, w21_t, b2_t]

    for mine, t in zip(mine_params, torch_params):
        assert t.grad is not None
        assert abs(mine.grad - t.grad.item()) < 1e-6
