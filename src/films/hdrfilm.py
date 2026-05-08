import drjit as dr
from src.core.types import Color3f

class Film:
	def __init__(self, H = 800, W = 800):
		self.H = H
		self.W = W
		self.aspect = W / H

		self.num_pixels = W * H
		self.img = Color3f(self.num_pixels)
