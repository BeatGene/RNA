"""Temporarily disable dropout during refinement, then restore training state."""
from functools import wraps


def evaluation_mode(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        states = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            return function(self, *args, **kwargs)
        finally:
            for module, state in states:
                module.training = state
    return wrapped
