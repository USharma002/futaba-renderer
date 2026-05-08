import numpy as np

from src.core.types import Float, Array2f


class Sampler:
	def __init__(self, sample_count=1, seed=0):
		self.sample_count = int(sample_count)
		self.seed = int(seed)
		self._rng = np.random.default_rng(self.seed)

	def sample_1d(self, n=1):
		values = self._rng.random(int(n), dtype=np.float32)
		return Float(values) if int(n) != 1 else Float(values[0])

	def sample_2d(self, n=1):
		values = self._rng.random((int(n), 2), dtype=np.float32)
		if int(n) == 1:
			return Array2f(values[0, 0], values[0, 1])
		return Array2f(values[:, 0], values[:, 1])


class IndependentSampler(Sampler):
	pass
