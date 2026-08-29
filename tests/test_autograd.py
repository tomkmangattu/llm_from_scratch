import torch

from llm.autograd import Value


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
