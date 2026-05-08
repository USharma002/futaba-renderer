import math

import drjit as dr

class Ray:
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction
        self.time = 0

    def __repr__(self):
    	return f"Ray({self.o}, {self.d})"
