from src.core.types import Array3f


class Shape:
	def __init__(self, bsdf_id=0, albedo=None):
		self.bsdf_id = bsdf_id
		self.albedo = Array3f(1.0, 1.0, 1.0) if albedo is None else Array3f(albedo)

	def bounds(self):
		raise NotImplementedError(f"{type(self).__name__} must implement bounds()")

	def ray_intersect(self, rays, active):
		raise NotImplementedError(f"{type(self).__name__} must implement ray_intersect()")
