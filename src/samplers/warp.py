import drjit as dr
from drjit.cuda import Float, Array2f, Array3f

class Warp:
    @staticmethod
    def square_to_uniform_square(sample):
        return sample

    @staticmethod
    def square_to_uniform_square_pdf(sample):
        valid = (sample.x >= 0.0) & (sample.x <= 1.0) & (sample.y >= 0.0) & (sample.y <= 1.0)
        return dr.select(valid, Float(1.0), Float(0.0))

    @staticmethod
    def _tent_inverse_cdf(u):
        return dr.select(u < 0.5, dr.sqrt(2.0 * u) - 1.0, 1.0 - dr.sqrt(2.0 * (1.0 - u)))

    @staticmethod
    def square_to_tent(sample):
        return Array2f(
            Warp._tent_inverse_cdf(sample.x),
            Warp._tent_inverse_cdf(sample.y)
        )

    @staticmethod
    def square_to_tent_pdf(p):
        valid = (p.x >= -1.0) & (p.x <= 1.0) & (p.y >= -1.0) & (p.y <= 1.0)
        pdf_val = (1.0 - dr.abs(p.x)) * (1.0 - dr.abs(p.y))
        return dr.select(valid, Float(pdf_val), Float(0.0))

    @staticmethod
    def square_to_uniform_disk(sample):
        r = dr.sqrt(sample.x)
        theta = 2.0 * dr.pi * sample.y
        return Array2f(r * dr.cos(theta), r * dr.sin(theta))

    @staticmethod
    def square_to_uniform_disk_pdf(p):
        dist_sq = p.x * p.x + p.y * p.y
        return dr.select(dist_sq <= 1.0, Float(1.0 / dr.pi), Float(0.0))

    @staticmethod
    def square_to_uniform_sphere(sample):
        z = sample.x * 2.0 - 1.0
        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
        phi = sample.y * 2.0 * dr.pi
        return Array3f(r * dr.cos(phi), r * dr.sin(phi), z)

    @staticmethod
    def square_to_uniform_sphere_pdf(v):
        # The PDF is uniform over the sphere surface
        n = dr.width(v) if dr.is_array_v(v) else 1
        return dr.full(Float, 1.0 / (4.0 * dr.pi), n)

    @staticmethod
    def square_to_uniform_hemisphere(sample):
        z = sample.x
        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
        phi = sample.y * 2.0 * dr.pi
        return Array3f(r * dr.cos(phi), r * dr.sin(phi), z)

    @staticmethod
    def square_to_uniform_hemisphere_pdf(v):
        valid = v.z >= 0.0
        return dr.select(valid, Float(1.0 / (2.0 * dr.pi)), Float(0.0))

    @staticmethod
    def square_to_cosine_hemisphere(sample):
        r = dr.sqrt(sample.x)
        theta = 2.0 * dr.pi * sample.y
        x = r * dr.cos(theta)
        y = r * dr.sin(theta)
        z = dr.sqrt(dr.maximum(1.0 - x * x - y * y, 0.0))
        return Array3f(x, y, z)

    @staticmethod
    def square_to_cosine_hemisphere_pdf(v):
        valid = v.z >= 0.0
        return dr.select(valid, Float(v.z / dr.pi), Float(0.0))

    @staticmethod
    def square_to_beckmann(sample, alpha=0.1):
        phi = 2.0 * dr.pi * sample.y
        tan2_theta = -alpha * alpha * dr.log(1.0 - sample.x)
        cos_theta = 1.0 / dr.sqrt(1.0 + tan2_theta)
        sin_theta = dr.sqrt(dr.maximum(1.0 - cos_theta * cos_theta, 0.0))
        x = sin_theta * dr.cos(phi)
        y = sin_theta * dr.sin(phi)
        z = cos_theta
        return Array3f(x, y, z)

    @staticmethod
    def square_to_beckmann_pdf(m, alpha=0.1):
        valid = m.z > 0.0
        cos_theta = dr.maximum(m.z, 1e-4)
        cos2_theta = cos_theta * cos_theta
        sin2_theta = dr.maximum(1.0 - cos2_theta, 0.0)
        tan2_theta = sin2_theta / cos2_theta
        
        alpha2 = alpha * alpha
        D = dr.exp(-tan2_theta / alpha2) / (dr.pi * alpha2 * cos2_theta * cos2_theta)
        
        # The PDF of sampling m is D(m) * cos_theta
        pdf_val = D * cos_theta
        return dr.select(valid, Float(pdf_val), Float(0.0))
