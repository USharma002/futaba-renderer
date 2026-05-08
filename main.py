from src.sensors.sensor import Sensor
import numpy as np

from src.films.hdrfilm import Film
from src.shapes.mesh import load_obj_mesh
from src.scene.scene import Scene
from src.integrators.path import Simple
from src.integrators.normals import Normals
from src.render.renderer import Renderer


film = Film(H=1024, W=1024)
pos = np.array([[0.0, 0.15, 1.2]])
target = np.array([[0.0, 0.0, 0.0]])

cam = Sensor(pos, target, 45, film=film)

bunny_path = r"scenes/bunny.obj"
bunny = load_obj_mesh(bunny_path, scale_to=0.5)
scene = Scene([bunny])
integrator = Normals({})

scene_dict = {
    'sensor': cam,
    'scene': scene,
    'integrator': integrator
}

renderer = Renderer(scene_dict)
img = renderer.render()
img = img.numpy().reshape(3, cam.film.H, cam.film.W)
import matplotlib.pyplot as plt
plt.imshow(img.transpose(1, 2, 0))
# plt.show()
plt.imsave("preview.png", img.transpose(1, 2, 0))
