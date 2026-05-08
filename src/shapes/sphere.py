import drjit as dr

from src.core.types import Array3f
from src.shapes.shape import Shape


class Sphere(Shape):
	def __init__(self, center, radius, bsdf_id=0, albedo=None):
		super().__init__(bsdf_id, albedo=albedo)
		self.c = Array3f(center)
		self.r = radius

	def __repr__(self):
		return f"Sphere(center={self.c}, radius={self.r})"

	def bounds(self):
		r = Array3f(self.r, self.r, self.r)
		return self.c - r, self.c + r

	def ray_intersect(self, rays, active):
		q = rays.o - self.c
		b = dr.dot(q, rays.d)
		c = dr.dot(q, q) - dr.square(self.r)
		discriminant = b * b - c

		has_roots = active & (discriminant >= 0.0)
		root = dr.sqrt(dr.maximum(discriminant, 0.0))
		t0 = -b - root
		t1 = -b + root
		t = dr.select(t0 > 1e-4, t0, t1)
		hit = has_roots & (t > 1e-4)

		p = rays.o + t * rays.d
		n = dr.select(hit, dr.normalize(p - self.c), dr.zeros(Array3f))
		return dr.select(hit, t, -1.0), n
