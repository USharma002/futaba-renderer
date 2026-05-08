import drjit as dr

try:
	Float = dr.cuda.Float
	Bool = dr.cuda.Bool
	Int = dr.cuda.Int
	Array2f = dr.cuda.Array2f
	Array3f = dr.cuda.Array3f
	Matrix4f = dr.cuda.Matrix4f
except Exception:
	Float = dr.llvm.Float
	Bool = dr.llvm.Bool
	Int = dr.llvm.Int
	Array2f = dr.llvm.Array2f
	Array3f = dr.llvm.Array3f

Vector3f = Array3f
Color3f = Array3f