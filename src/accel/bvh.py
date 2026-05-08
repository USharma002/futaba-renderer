from dataclasses import dataclass

import drjit as dr
import numpy as np

from src.core.frame import Frame
from src.core.types import Array3f, Float, Int
from src.integrators.surface_interaction import SurfaceInteraction
from src.shapes.mesh import Mesh


@dataclass
class BVHNode:
	bmin: np.ndarray
	bmax: np.ndarray
	left: int = -1
	right: int = -1
	indices: list[int] | None = None

	@property
	def is_leaf(self):
		return self.left < 0


@dataclass
class MeshGPUData:
	tri_v0: Array3f
	tri_e1: Array3f
	tri_e2: Array3f
	tri_normal: Array3f
	tri_albedo: Array3f
	node_bmin: Array3f
	node_bmax: Array3f
	node_left: Int
	node_right: Int
	leaf_start: Int
	leaf_count: Int
	leaf_indices: Int

	def eval(self):
		dr.eval(
			self.tri_v0,
			self.tri_e1,
			self.tri_e2,
			self.tri_normal,
			self.tri_albedo,
			self.node_bmin,
			self.node_bmax,
			self.node_left,
			self.node_right,
			self.leaf_start,
			self.leaf_count,
			self.leaf_indices,
		)


class BVHAccel:
	def __init__(self, shapes, leaf_size=4):
		self.shapes = list(shapes)
		self.leaf_size = max(1, int(leaf_size))
		self.mesh = self._merge_meshes(self.shapes) if self.shapes and all(isinstance(shape, Mesh) for shape in self.shapes) else None
		self.nodes = []

		if self.mesh is not None:
			self.root = self._build(list(range(self.mesh.triangle_count)), self.mesh.tri_bmin_np, self.mesh.tri_bmax_np)
		else:
			bounds = [self._shape_bounds(shape) for shape in self.shapes]
			bmin = np.asarray([bound[0] for bound in bounds], dtype=np.float32) if bounds else np.zeros((0, 3), dtype=np.float32)
			bmax = np.asarray([bound[1] for bound in bounds], dtype=np.float32) if bounds else np.zeros((0, 3), dtype=np.float32)
			self.root = self._build(list(range(len(self.shapes))), bmin, bmax) if self.shapes else -1

		self.max_leaf_size = max((len(node.indices) for node in self.nodes if node.is_leaf), default=self.leaf_size)
		self.max_stack_depth = max(32, self._depth(self.root) + 2)
		self.gpu_data = self._build_mesh_gpu_data() if self.mesh is not None else None
		if self.gpu_data is not None:
			self.gpu_data.eval()

	def ray_intersect(self, rays, active):
		if self.root < 0:
			return self._empty_interaction(rays, active)
		if self.mesh is not None:
			return self._intersect_mesh(rays, active)
		return self._intersect_shapes(rays, active)

	def _merge_meshes(self, shapes):
		return shapes[0] if len(shapes) == 1 else Mesh.concat(shapes)

	def _shape_bounds(self, shape):
		if hasattr(shape, "bounds_np"):
			return shape.bounds_np
		bmin, bmax = shape.bounds()
		return (
			np.asarray(bmin.numpy(), dtype=np.float32).reshape(-1)[:3],
			np.asarray(bmax.numpy(), dtype=np.float32).reshape(-1)[:3],
		)

	def _build(self, indices, primitive_bmin, primitive_bmax):
		node_bmin = primitive_bmin[indices].min(axis=0)
		node_bmax = primitive_bmax[indices].max(axis=0)
		node_index = len(self.nodes)
		self.nodes.append(BVHNode(node_bmin, node_bmax, indices=indices))

		if len(indices) <= self.leaf_size:
			return node_index

		centroids = 0.5 * (primitive_bmin[indices] + primitive_bmax[indices])
		extent = centroids.max(axis=0) - centroids.min(axis=0)
		split_axis = int(np.argmax(extent))
		if extent[split_axis] <= 0.0:
			return node_index

		order = np.argsort(centroids[:, split_axis], kind="mergesort")
		mid = len(indices) // 2
		left_indices = [indices[i] for i in order[:mid]]
		right_indices = [indices[i] for i in order[mid:]]
		if not left_indices or not right_indices:
			return node_index

		node = self.nodes[node_index]
		node.left = self._build(left_indices, primitive_bmin, primitive_bmax)
		node.right = self._build(right_indices, primitive_bmin, primitive_bmax)
		node.indices = None
		return node_index

	def _depth(self, node_index):
		if node_index < 0:
			return 0
		node = self.nodes[node_index]
		if node.is_leaf:
			return 1
		return 1 + max(self._depth(node.left), self._depth(node.right))

	def _array3f(self, values):
		values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
		return Array3f(values[:, 0], values[:, 1], values[:, 2])

	def _build_mesh_gpu_data(self):
		node_bmin = []
		node_bmax = []
		node_left = []
		node_right = []
		leaf_start = []
		leaf_count = []
		leaf_indices = []

		for node in self.nodes:
			node_bmin.append(node.bmin)
			node_bmax.append(node.bmax)
			node_left.append(node.left)
			node_right.append(node.right)

			if node.is_leaf:
				leaf_start.append(len(leaf_indices))
				leaf_count.append(len(node.indices))
				leaf_indices.extend(node.indices)
			else:
				leaf_start.append(0)
				leaf_count.append(0)

		if not leaf_indices:
			leaf_indices = [0]

		return MeshGPUData(
			tri_v0=self._array3f(self.mesh.tri_v0_np),
			tri_e1=self._array3f(self.mesh.tri_e1_np),
			tri_e2=self._array3f(self.mesh.tri_e2_np),
			tri_normal=self._array3f(self.mesh.tri_normal_np),
			tri_albedo=self._array3f(self.mesh.tri_albedo_np),
			node_bmin=self._array3f(node_bmin),
			node_bmax=self._array3f(node_bmax),
			node_left=Int(np.asarray(node_left, dtype=np.int32)),
			node_right=Int(np.asarray(node_right, dtype=np.int32)),
			leaf_start=Int(np.asarray(leaf_start, dtype=np.int32)),
			leaf_count=Int(np.asarray(leaf_count, dtype=np.int32)),
			leaf_indices=Int(np.asarray(leaf_indices, dtype=np.int32)),
		)

	def _ray_width(self, rays, active):
		width = max(dr.width(rays.o), dr.width(rays.d))
		try:
			width = max(width, dr.width(active))
		except Exception:
			pass
		return width

	def _empty_interaction(self, rays, active):
		return self._finish_interaction(SurfaceInteraction(self._ray_width(rays, active)), rays)

	def _finish_interaction(self, si, rays):
		si.p = rays.o + si.t * rays.d
		si.frame = Frame(si.n)
		si.wi = si.to_local(-rays.d)
		si.sh_normal = si.to_local(si.n)
		return si

	def _ray_aabb_intersect(self, rays, box_min, box_max):
		def slab(origin, direction, lower, upper):
			parallel = dr.abs(direction) < 1e-12
			outside = parallel & ((origin < lower) | (origin > upper))
			safe_direction = dr.select(parallel, 1.0, direction)
			t0 = (lower - origin) / safe_direction
			t1 = (upper - origin) / safe_direction
			return t0, t1, outside

		t0x, t1x, outside_x = slab(rays.o.x, rays.d.x, box_min.x, box_max.x)
		t0y, t1y, outside_y = slab(rays.o.y, rays.d.y, box_min.y, box_max.y)
		t0z, t1z, outside_z = slab(rays.o.z, rays.d.z, box_min.z, box_max.z)

		tmin = dr.maximum(dr.maximum(dr.minimum(t0x, t1x), dr.minimum(t0y, t1y)), dr.minimum(t0z, t1z))
		tmax = dr.minimum(dr.minimum(dr.maximum(t0x, t1x), dr.maximum(t0y, t1y)), dr.maximum(t0z, t1z))
		hit = ~(outside_x | outside_y | outside_z) & (tmax >= dr.maximum(tmin, 0.0))
		return hit, tmin, tmax

	def _ray_aabb_intersect_node(self, rays, node):
		box_min = Array3f(float(node.bmin[0]), float(node.bmin[1]), float(node.bmin[2]))
		box_max = Array3f(float(node.bmax[0]), float(node.bmax[1]), float(node.bmax[2]))
		return self._ray_aabb_intersect(rays, box_min, box_max)

	def _triangle_intersect(self, rays, v0, e1, e2, active):
		pvec = dr.cross(rays.d, e2)
		det = dr.dot(e1, pvec)
		det_valid = dr.abs(det) > 1e-8
		inv_det = dr.select(det_valid, 1.0 / det, 0.0)
		svec = rays.o - v0
		u = dr.dot(svec, pvec) * inv_det
		qvec = dr.cross(svec, e1)
		v = dr.dot(rays.d, qvec) * inv_det
		t = dr.dot(e2, qvec) * inv_det
		hit = active & det_valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4)
		return t, hit

	def _detach_mesh_state(self, current, stack_ptr, stack, mint, si):
		current = dr.detach(current)
		stack_ptr = dr.detach(stack_ptr)
		stack = dr.detach(stack)
		mint = dr.detach(mint)
		si.t = dr.detach(si.t)
		si.n = dr.detach(si.n)
		si.albedo = dr.detach(si.albedo)
		si.shape_id = dr.detach(si.shape_id)
		dr.eval(current, stack_ptr, stack, mint, si.t, si.n, si.albedo, si.shape_id)
		return current, stack_ptr, stack, mint, si

	def _intersect_mesh(self, rays, active):
		data = self.gpu_data
		n_rays = self._ray_width(rays, active)
		si = SurfaceInteraction(n_rays)
		mint = dr.full(Float, float("inf"), n_rays)

		current = dr.full(Int, self.root, n_rays)
		stack_ptr = dr.zeros(Int, n_rays)
		stack = dr.full(Int, -1, n_rays * self.max_stack_depth)
		ray_index = dr.arange(Int, n_rays)
		max_iterations = max(1, len(self.nodes) * 2)

		for _ in range(max_iterations):
			lane_active = dr.detach(active & (current >= 0))
			dr.eval(lane_active)
			if not dr.any(lane_active):
				break

			node_index = dr.maximum(current, 0)
			box_min = dr.gather(Array3f, data.node_bmin, node_index, lane_active)
			box_max = dr.gather(Array3f, data.node_bmax, node_index, lane_active)
			left = dr.gather(Int, data.node_left, node_index, lane_active)
			right = dr.gather(Int, data.node_right, node_index, lane_active)
			leaf_start = dr.gather(Int, data.leaf_start, node_index, lane_active)
			leaf_count = dr.gather(Int, data.leaf_count, node_index, lane_active)

			node_hit, tmin, _ = self._ray_aabb_intersect(rays, box_min, box_max)
			node_hit = lane_active & node_hit & (tmin < mint)
			is_leaf = node_hit & (left < 0)
			is_internal = node_hit & (left >= 0)

			for leaf_offset in range(self.max_leaf_size):
				triangle_active = is_leaf & (leaf_offset < leaf_count)
				triangle_index = dr.gather(Int, data.leaf_indices, leaf_start + leaf_offset, triangle_active)
				v0 = dr.gather(Array3f, data.tri_v0, triangle_index, triangle_active)
				e1 = dr.gather(Array3f, data.tri_e1, triangle_index, triangle_active)
				e2 = dr.gather(Array3f, data.tri_e2, triangle_index, triangle_active)
				t_hit, shape_hit = self._triangle_intersect(rays, v0, e1, e2, triangle_active)
				shape_hit = shape_hit & (t_hit < mint)

				n_hit = dr.gather(Array3f, data.tri_normal, triangle_index, shape_hit)
				albedo = dr.gather(Array3f, data.tri_albedo, triangle_index, shape_hit)
				si.t = dr.select(shape_hit, t_hit, si.t)
				si.n = dr.select(shape_hit, n_hit, si.n)
				si.albedo = dr.select(shape_hit, albedo, si.albedo)
				si.shape_id = dr.select(shape_hit, triangle_index, si.shape_id)
				mint = dr.select(shape_hit, t_hit, mint)

			can_push = is_internal & (stack_ptr < self.max_stack_depth)
			push_index = ray_index * self.max_stack_depth + stack_ptr
			dr.scatter(stack, right, push_index, can_push)
			pushed_ptr = dr.select(can_push, stack_ptr + 1, stack_ptr)

			pop_active = lane_active & ~is_internal
			has_stack = pushed_ptr > 0
			pop_index = ray_index * self.max_stack_depth + pushed_ptr - 1
			popped = dr.gather(Int, stack, dr.maximum(pop_index, 0), pop_active & has_stack)
			popped = dr.select(pop_active & has_stack, popped, -1)
			popped_ptr = dr.select(pop_active & has_stack, pushed_ptr - 1, pushed_ptr)

			current = dr.select(is_internal, left, popped)
			stack_ptr = dr.select(is_internal, pushed_ptr, popped_ptr)
			current, stack_ptr, stack, mint, si = self._detach_mesh_state(current, stack_ptr, stack, mint, si)
		else:
			raise RuntimeError("BVH traversal exceeded its iteration budget")

		return self._finish_interaction(si, rays)

	def _detach_shape_state(self, mint, si):
		mint = dr.detach(mint)
		si.t = dr.detach(si.t)
		si.n = dr.detach(si.n)
		si.albedo = dr.detach(si.albedo)
		si.shape_id = dr.detach(si.shape_id)
		dr.eval(mint, si.t, si.n, si.albedo, si.shape_id)
		return mint, si

	def _intersect_shapes(self, rays, active):
		n_rays = self._ray_width(rays, active)
		si = SurfaceInteraction(n_rays)
		mint = dr.full(Float, float("inf"), n_rays)
		stack = [self.root]

		while stack:
			node = self.nodes[stack.pop()]
			hit, tmin, _ = self._ray_aabb_intersect_node(rays, node)
			node_active = dr.detach(active & hit & (tmin < mint))
			dr.eval(node_active)
			if not dr.any(node_active):
				continue

			if node.is_leaf:
				for shape_index in node.indices:
					shape = self.shapes[shape_index]
					t_hit, n_hit = shape.ray_intersect(rays, node_active)
					shape_hit = node_active & (t_hit > 0.0) & (t_hit < mint)
					si.t = dr.select(shape_hit, t_hit, si.t)
					si.n = dr.select(shape_hit, n_hit, si.n)
					si.albedo = dr.select(shape_hit, shape.albedo, si.albedo)
					si.shape_id = dr.select(shape_hit, shape_index, si.shape_id)
					mint = dr.select(shape_hit, t_hit, mint)
				mint, si = self._detach_shape_state(mint, si)
			else:
				stack.append(node.right)
				stack.append(node.left)

		return self._finish_interaction(si, rays)
