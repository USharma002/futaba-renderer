import drjit as dr

from src.core.types import Float, Vector3f, Color3f, Bool, Int

class BSDFContext:
    def __init__(self):
        # In Mitsuba, this holds TransportMode (Radiance vs Importance), 
        # but for a basic path tracer, an empty context is fine!
        pass

class BSDFSample:
    def __init__(self, n_rays):
        self.wo = dr.zeros(Vector3f, n_rays)  # The sampled outgoing direction (Local Space)
        self.pdf = dr.zeros(Float, n_rays)    # The probability of this sample
        self.eta = dr.full(Float, 1.0, n_rays)# Index of refraction (for glass)
        self.sampled_type = dr.zeros(Int, n_rays) # e.g., Diffuse, Specular, etc.

class BSDF:
    def __init__(self):
        pass

    def eval(self, ctx, si, wo, active):
        """
        Evaluates the BSDF f(wi, wo) * cos(theta_wo).
        Returns the Color/Spectrum.
        """
        pass

    def sample(self, ctx, si, sample1, sample2, active):
        """
        Takes 2 random numbers (sample1, sample2).
        Returns a tuple: (BSDFSample, BSDF_Weight)
        Where BSDF_Weight = eval() / pdf
        """
        pass

    def pdf(self, ctx, si, wo, active):
        """
        Returns the Probability Density Function for a given wi and wo.
        """
        pass