import drjit as dr

from src.core.types import Array3f
from src.shapes.shape import Shape


class Triangle(Shape):
	def __init__(self, v0, v1, v2, bsdf_id=0, albedo=None):
		super().__init__(bsdf_id, albedo=albedo)
		self.v0 = Array3f(v0)
		self.v1 = Array3f(v1)
		self.v2 = Array3f(v2)
		self.e1 = self.v1 - self.v0
		self.e2 = self.v2 - self.v0
		self.n = dr.normalize(dr.cross(self.e1, self.e2))

	def bounds(self):
		return (
			dr.minimum(dr.minimum(self.v0, self.v1), self.v2),
			dr.maximum(dr.maximum(self.v0, self.v1), self.v2),
		)

	def ray_intersect(self, rays, active):
		pvec = dr.cross(rays.d, self.e2)
		det = dr.dot(self.e1, pvec)
		det_valid = dr.abs(det) > 1e-8
		inv_det = dr.select(det_valid, 1.0 / det, 0.0)

		svec = rays.o - self.v0
		u = dr.dot(svec, pvec) * inv_det
		qvec = dr.cross(svec, self.e1)
		v = dr.dot(rays.d, qvec) * inv_det
		t = dr.dot(self.e2, qvec) * inv_det

		hit = (
			active &
			det_valid &
			(u >= 0.0) & (u <= 1.0) &
			(v >= 0.0) & (u + v <= 1.0) &
			(t > 1e-4)
		)
		return dr.select(hit, t, -1.0), dr.select(hit, self.n, dr.zeros(Array3f))
