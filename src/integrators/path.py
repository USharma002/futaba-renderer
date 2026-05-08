import drjit as dr
from src.core.types import Array3f

class Simple:
	def __init__(self, props):
		self.max_depth = props.get('max_depth', 5)
		self.rr_depth = props.get('rr_depth', 5)

	@dr.syntax
	def sample(self, scene, sampler, rays, active):
		si = scene.ray_intersect(rays, active)
		light_dir = dr.normalize(Array3f(1.0, 1.0, 1.0))

		cos_theta = dr.maximum(dr.dot(si.n, light_dir), 0.0)
		background = Array3f(0.05, 0.05, 0.05)
		L = dr.select(si.is_valid(), si.albedo * cos_theta, background)

		return (L, si.is_valid(), [])