"""One-step numerical integrators. velocity(position, graph_time) -> [N,3]."""
import torch
from etflow.models.utils import center_of_mass


def euler(x, velocity, dt, **kwargs):
    return x + dt * velocity


def heun(x, velocity, dt, evaluate, next_time, **kwargs):
    return x + 0.5 * dt * (velocity + evaluate(x + dt * velocity, next_time))


def euler_maruyama(x, velocity, dt, batch, std, s_churn, **kwargs):
    noise = center_of_mass(torch.randn_like(x), batch=batch)
    return x + dt * velocity + std * s_churn * dt.sqrt() * noise


SOLVERS = {"ode": euler, "heun": heun, "sde": euler_maruyama}
