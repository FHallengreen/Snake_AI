import random
from typing import Protocol, Tuple, List, Sequence
import numpy as np
from ga_models.ga_protocol import GAModel
from ga_models.activation import relu, softmax


class SimpleModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self._DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                self._DNA.append(np.random.normal(0, 0.1, (dim, dims[i+1])))

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = np.array(obs)
        for i, layer in enumerate(self._DNA):
            x = relu(x) if i > 0 else x
            x = x @ layer
        return softmax(x)

    def action(self, obs: Sequence):
        return self.update(obs).argmax()

    def mutate(self, mutation_rate) -> None:
        for layer in self._DNA:
            mask = np.random.random(layer.shape) < mutation_rate
            layer[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

    def __add__(self, other):
        baby_DNA = []
        alpha = random.uniform(0.4, 0.6)
        for mom, dad in zip(self._DNA, other._DNA):
            baby_layer = mom * alpha + dad * (1 - alpha)
            baby_DNA.append(baby_layer.copy())
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    @property
    def DNA(self):
        return self._DNA

    @DNA.setter
    def DNA(self, value):
        self._DNA = value