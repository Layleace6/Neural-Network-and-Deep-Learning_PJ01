import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Base class for all optimizers.
    """
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr  # Initial learning rate
        self.lr = init_lr       # Current learning rate (can be updated by schedulers)
        self.model = model      # Model to be optimized

    @abstractmethod
    def step(self) -> None:
        """
        Perform a single optimization step (parameter update).
        """
        pass

    def _apply_weight_decay(self, layer, key) -> None:
        """
        Apply L2 weight decay regularization to a given parameter if needed.
        """
        if hasattr(layer, "weight_decay") and layer.weight_decay:
            layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)


class SGD(Optimizer):
    """
    Standard Stochastic Gradient Descent optimizer.
    """
    def __init__(self, init_lr, model) -> None:
        super().__init__(init_lr, model)

    def step(self) -> None:
        for layer in self.model.layers:
            if hasattr(layer, "optimizable") and layer.optimizable:
                for key in layer.params.keys():
                    self._apply_weight_decay(layer, key)
                    # Standard SGD update rule: param <- param - lr * grad
                    layer.params[key] -= self.lr * layer.grads[key]


class MomentGD(Optimizer):
    """
    Stochastic Gradient Descent with Momentum optimizer.
    """
    def __init__(self, init_lr, model, mu=0.9) -> None:
        super().__init__(init_lr, model)
        self.mu = mu  # Momentum coefficient
        self.velocities = {}

        # Initialize velocities to zero arrays matching parameter shapes
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, "optimizable") and layer.optimizable:
                self.velocities[i] = {}
                for key in layer.params.keys():
                    self.velocities[i][key] = np.zeros_like(layer.params[key])

    def step(self) -> None:
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, "optimizable") and layer.optimizable:
                for key in layer.params.keys():
                    self._apply_weight_decay(layer, key)
                    v = self.velocities[i][key]
                    g = layer.grads[key]

                    # Update velocity: v <- mu * v - lr * g
                    v = self.mu * v - self.lr * g
                    # Update parameter: param <- param + v
                    layer.params[key] += v
                    # Store updated velocity
                    self.velocities[i][key] = v
