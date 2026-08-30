from llm.autograd import MLP, Value


def train_using_autograd():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP(3, [4, 4, 1])

    step = 0
    lr = 0.05
    while step <= 100:

        ypred = [mlp(x) for x in xs]

        for params in  mlp.parameters():
            params.grad = 0.0

        loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), start=Value(0.0))

        loss.backward()

        for params in  mlp.parameters():
            params.data -= lr * params.grad

        if step == 0 or step % 10 == 0:
            print(f'step {step} loss {loss}')
            prediction = [ round(y.data,5) for y in ypred]
            print(f'prediction {prediction}')

        step += 1

train_using_autograd()