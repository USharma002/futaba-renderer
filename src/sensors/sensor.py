import math

import drjit as dr
from src.films.hdrfilm import Film
from src.core.ray import Ray

class Sensor:
	def __init__(self, pos, target, fov, film = None):
		self.pos = dr.cuda.Array3f(*pos)
		self.target = dr.cuda.Array3f(*target)
		self.world_up = dr.cuda.Array3f(0, 1, 0)

		self.film = film if film is not None else Film()

		self.f = dr.cuda.Float(1.0)
		self.fov = fov


	def lookAt(self, pos, target):
		forward = dr.normalize(target - pos)
		right = dr.normalize(dr.cross(forward, self.world_up))
		up = dr.normalize(dr.cross(right, forward))

		M_cam = dr.cuda.Matrix3f(right, up, forward)
		return M_cam

	def sample(self):
		H, W = self.film.H, self.film.W
		aspect = self.film.aspect

		num_pixels = self.film.num_pixels

		idx = dr.arange(dr.cuda.Int, num_pixels)

		x_indx = idx % self.film.W
		y_indx = idx // self.film.W

		x_center = (x_indx + 0.5) / W
		y_center = (y_indx + 0.5) / H

		xs = x_center * 2.0 - 1.0
		ys = y_center * -2.0 + 1.0

		height = 2.0 * self.f * math.tan(math.radians(self.fov) / 2.0)
		width = height * aspect

		px = xs * (width / 2.0)
		py = ys * (height / 2.0)

		M_cam = self.lookAt(self.pos, self.target)

		dirs_cam = dr.cuda.Array3f(px, py, self.f)

		rd = M_cam @ dirs_cam
		rd = dr.normalize(rd)

		ro = self.pos

		return Ray(ro, rd)