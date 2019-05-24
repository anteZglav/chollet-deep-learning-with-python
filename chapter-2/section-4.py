from matplotlib import pyplot as plt
# 2.4.3
# Implement gradient descent with momentum
class OptimizationProblem:
    """Class for optimization problem."""

    def __init__(self, func, gradient_func, W):
        """"Initialize the problem.
            func = f(W)- must be diferentiable.
            gradient_func = f'(W) - gradient function of f.
                """
        assert callable(func)
        assert callable(gradient_func)
        self.function = func
        self.gradient_function = gradient_func
        self.W = W

    def get_loss(self):
        return self.function(self.W)

    def get_gradient(self):
        return self.gradient_function(self.W)

    def get_current_paramters(self):
        return self.W, self.get_loss(), self.get_gradient()

    def update_parameter(self, W):
        self.W = W


if __name__ == '__main__':
    def f(x): return (x) ** 2 + 5*x

    def df(x): return 2 * x + 5

    problem = OptimizationProblem(f,df, 5)
    learning_rate = 0.1
    loss = problem.get_loss()

    past_velocity = 0.
    momentum = 0.1
    while loss > 0.01:
        W, loss, gradient = problem.get_current_paramters()
        velocity = past_velocity * momentum + learning_rate * gradient
        W = W + momentum * past_velocity - learning_rate * gradient
        past_velocity = velocity
        problem.update_parameter(W)
        print(f"loss = {loss}")
