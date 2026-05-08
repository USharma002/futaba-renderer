from src.accel import BVHAccel


class Scene:
	def __init__(self, shapes, leaf_size=4):
		self.shapes = list(shapes)
		self.accel = BVHAccel(self.shapes, leaf_size=leaf_size)

	def ray_intersect(self, rays, active):
		return self.accel.ray_intersect(rays, active)
