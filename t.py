import matplotlib.pyplot as plt

from optics import *


if __name__ =='__main__':
    _slm = SLM()
    _camera = Camera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (3, 1), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=10)
    sim.register()

    result = sim.propagate(sim.holo_box(_tr.numba_holo_traps([1.1411255,  1.11089067, 0.74798383])))
    v = sim.check_intensities(result)
    print(1 - (max(v) - min(v)) / (max(v) + min(v)))
    im = plt.imshow(result, cmap='nipy_spectral')
    plt.colorbar(im)
    plt.show()