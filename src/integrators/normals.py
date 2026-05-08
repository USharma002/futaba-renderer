import drjit as dr

from src.core.types import Array3f


class Normals:
	def __init__(self, props):
		self.props = props

	@dr.syntax
	def sample(self, scene, sampler, rays, active):
		si = scene.ray_intersect(rays, active)
		color = dr.select(si.is_valid(), si.n * 0.5 + 0.5, Array3f(0.0, 0.0, 0.0))
		return color, si.is_valid(), []
