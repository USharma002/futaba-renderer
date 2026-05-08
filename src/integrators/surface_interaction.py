import drjit as dr
from drjit.cuda import Float, Array2f, Array3f, Int

from src.core.frame import Frame

class SurfaceInteraction:
	def __init__(self, n_rays):
		# 1. Geometry (World Space)
		self.t = dr.full(Float, float('inf'), n_rays)
		self.p = dr.zeros(Array3f, n_rays)
		self.n = dr.zeros(Array3f, n_rays) # Geometric normal
		self.albedo = dr.zeros(Array3f, n_rays)

		# Shading frame
		self.sh_frame_n = dr.zeros(Array3f, n_rays)
		self.sh_frame_s = dr.zeros(Array3f, n_rays) # Tangent
		self.sh_frame_t = dr.zeros(Array3f, n_rays) # Bitangent

		# incoming direction
		self.wi = dr.zeros(Array3f, n_rays)

		self.uv = dr.zeros(Array2f, n_rays)
		self.shape_id = dr.full(Int, -1, n_rays)

		self.frame = Frame(self.n)

	def is_valid(self):
		return dr.isfinite(self.t)

	def to_world(self, dirs):
		return self.frame.to_world(dirs)

	def to_local(self, dirs):
		return self.frame.to_local(dirs)