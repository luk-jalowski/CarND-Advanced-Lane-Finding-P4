import numpy as np

class Camera():
    def __init__(self):
        self.mtx = 0
        self.dist = []
        self.source = np.float32([])
        self.destination = np.float32([])