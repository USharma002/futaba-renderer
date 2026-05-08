import drjit as dr
from src.core.types import Float, Array3f

class Frame:
	def __init__(self, n):
		self.n = dr.normalize(n)

		sign = dr.select(self.n.z >= 0.0, 1.0, -1.0)
		a = -1.0 / (sign + self.n.z)
		b = self.n.x * self.n.y * a

		self.s = Array3f(1.0 + sign * self.n.x * self.n.x * a, sign * b, -sign * self.n.x)
		self.t = Array3f(b, sign + self.n.y * self.n.y * a, -self.n.y)

	def to_local(self, v):
		ArrayType = type(self.n)
		return ArrayType(dr.dot(v, self.s), dr.dot(v, self.t), dr.dot(v, self.n))

	def to_world(self, v):
		return self.s * v.x + self.t * v.y + self.n * v.z
