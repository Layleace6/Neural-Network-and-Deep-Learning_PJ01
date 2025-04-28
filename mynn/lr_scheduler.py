from abc import abstractmethod

class Scheduler:
    """
    Base class for all learning rate schedulers.
    """
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer  # Optimizer to adjust learning rate for
        self.step_count = 0          # Counter for steps taken
    
    @abstractmethod
    def step(self) -> None:
        """
        Update the learning rate according to the scheduling policy.
        """
        pass

    def get_lr(self):
        """
        Return the current learning rate.
        """
        return self.optimizer.init_lr


class StepLR(Scheduler):
    """
    StepLR scheduler: Multiply learning rate by gamma every step_size steps.
    """
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0


class MultiStepLR(Scheduler):
    """
    MultiStepLR scheduler: Multiply learning rate by gamma at specific milestones.
    """
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  # Sort milestones in ascending order
        self.gamma = gamma
        self.current_milestone_index = 0

    def step(self) -> None:
        self.step_count += 1
        if (self.current_milestone_index < len(self.milestones) and
            self.step_count == self.milestones[self.current_milestone_index]):
            self.optimizer.init_lr *= self.gamma
            self.current_milestone_index += 1


class ExponentialLR(Scheduler):
    """
    ExponentialLR scheduler: Multiply learning rate by gamma every step.
    """
    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma


class MomentumLR(Scheduler):
    """
    Momentum-based LR scheduler:
    Learning rate change depends on both current decay and momentum from previous change.
    """
    def __init__(self, optimizer, gamma=0.95, beta=0.9) -> None:
        super().__init__(optimizer)
        self.gamma = gamma  # Decay factor
        self.beta = beta    # Momentum factor
        self.prev_lr_delta = 0  # Previous learning rate change (initially zero)

    def step(self) -> None:
        self.step_count += 1
        # Current decay term
        current_delta = (self.optimizer.init_lr * (self.gamma - 1))
        # Add momentum from previous change
        total_delta = current_delta + self.beta * self.prev_lr_delta
        # Update learning rate
        self.optimizer.init_lr += total_delta
        # Store this step's delta for next step
        self.prev_lr_delta = total_delta
