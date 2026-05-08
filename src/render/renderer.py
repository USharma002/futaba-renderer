import drjit as dr

class Renderer:
	def __init__(self, props):
		self.sensor = props.get('sensor', None)
		self.integrator = props.get('integrator', None)
		self.sampler = props.get('sampler', None)
		self.scene = props.get('scene', None)

	def render(self):
		rays = self.sensor.sample()
		active = dr.cuda.Bool(True)
		sampler = self.sampler
		if sampler is None:
			from src.samplers.sampler import IndependentSampler
			sampler = IndependentSampler()

		L, active, aovs = self.integrator.sample(self.scene, sampler, rays, active)

		dr.eval(L)

		return L