import numpy as np


class DiscreteDistribution1D:
	def __init__(self, weights):
		self.weights = np.asarray(weights, dtype=np.float64)
		if self.weights.ndim != 1:
			raise ValueError("weights must be one-dimensional")
		self.total = float(self.weights.sum())
		if self.total > 0.0:
			self.pdf = self.weights / self.total
			self.cdf = np.concatenate(([0.0], np.cumsum(self.pdf)))
			self.cdf[-1] = 1.0
		else:
			self.pdf = np.zeros_like(self.weights)
			self.cdf = np.linspace(0.0, 1.0, len(self.weights) + 1)

	def sample(self, u):
		u = float(u)
		idx = int(np.searchsorted(self.cdf, u, side="right") - 1)
		idx = max(0, min(idx, len(self.weights) - 1))
		return idx, float(self.pdf[idx])

	def pmf(self, index):
		return float(self.pdf[int(index)])


class Distribution1D(DiscreteDistribution1D):
	pass
