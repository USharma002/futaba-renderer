import numpy as np

from src.core.types import Array3f
from src.shapes.shape import Shape


class Mesh(Shape):
	def __init__(self, vertices, indices, bsdf_id=0, albedo=None):
		super().__init__(bsdf_id, albedo=albedo)
		self.albedo_value = [1.0, 1.0, 1.0] if albedo is None else albedo
		self.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
		self.indices = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
		self._build_triangle_data()

	@property
	def triangle_count(self):
		return len(self.indices)

	def _build_triangle_data(self):
		if self.triangle_count == 0:
			empty_vec = np.zeros((0, 3), dtype=np.float32)
			self.tri_v0_np = empty_vec
			self.tri_e1_np = empty_vec
			self.tri_e2_np = empty_vec
			self.tri_normal_np = empty_vec
			self.tri_albedo_np = empty_vec
			self.tri_bmin_np = empty_vec
			self.tri_bmax_np = empty_vec
			self.bounds_np = (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
			return

		v0 = self.vertices[self.indices[:, 0]]
		v1 = self.vertices[self.indices[:, 1]]
		v2 = self.vertices[self.indices[:, 2]]
		e1 = v1 - v0
		e2 = v2 - v0
		normals = np.cross(e1, e2)
		lengths = np.linalg.norm(normals, axis=1)
		valid = lengths > 0.0
		normals[valid] /= lengths[valid, None]
		normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

		self.tri_v0_np = v0
		self.tri_e1_np = e1
		self.tri_e2_np = e2
		self.tri_normal_np = normals.astype(np.float32, copy=False)
		albedo = np.asarray(self.albedo_value, dtype=np.float32).reshape(-1)[:3]
		self.tri_albedo_np = np.repeat(albedo[None, :], self.triangle_count, axis=0)
		self.tri_bmin_np = np.minimum(np.minimum(v0, v1), v2)
		self.tri_bmax_np = np.maximum(np.maximum(v0, v1), v2)
		self.bounds_np = (self.vertices.min(axis=0), self.vertices.max(axis=0))

	def bounds(self):
		bmin, bmax = self.bounds_np
		return (
			Array3f(float(bmin[0]), float(bmin[1]), float(bmin[2])),
			Array3f(float(bmax[0]), float(bmax[1]), float(bmax[2])),
		)

	@classmethod
	def concat(cls, meshes):
		vertices = []
		indices = []
		albedos = []
		vertex_offset = 0

		for mesh in meshes:
			vertices.append(mesh.vertices)
			indices.append(mesh.indices + vertex_offset)
			albedos.append(mesh.tri_albedo_np)
			vertex_offset += len(mesh.vertices)

		merged = cls(np.vstack(vertices), np.vstack(indices))
		merged.tri_albedo_np = np.vstack(albedos).astype(np.float32, copy=False)
		return merged


def _parse_obj_vertex_index(token, vertex_count):
	index_text = token.split("/")[0]
	index = int(index_text)
	return vertex_count + index if index < 0 else index - 1


def load_obj_mesh(path, bsdf_id=0, albedo=None, center=True, scale_to=0.5):
	vertices = []
	triangles = []

	with open(path, "r", encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if line.startswith("v "):
				_, x, y, z = line.split()[:4]
				vertices.append([float(x), float(y), float(z)])
			elif line.startswith("f "):
				tokens = line.split()[1:]
				if len(tokens) < 3:
					continue
				face = [_parse_obj_vertex_index(token, len(vertices)) for token in tokens]
				for i in range(1, len(face) - 1):
					triangles.append([face[0], face[i], face[i + 1]])

	if not vertices:
		return Mesh([], [], bsdf_id=bsdf_id, albedo=albedo)

	vertex_array = np.asarray(vertices, dtype=np.float32)
	if center:
		vertex_array = vertex_array - 0.5 * (vertex_array.min(axis=0) + vertex_array.max(axis=0))

	if scale_to is not None:
		extent = vertex_array.max(axis=0) - vertex_array.min(axis=0)
		vertex_array *= float(scale_to) / max(float(np.max(extent)), 1e-8)

	return Mesh(vertex_array, np.asarray(triangles, dtype=np.int32), bsdf_id=bsdf_id, albedo=albedo)
