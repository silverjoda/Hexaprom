import numpy as np

class PDreg:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = None

    def update(self, input, reference):
        assert len(input) == len(reference)
        error = np.array(reference) - np.array(input)
        K_ctrl = self.kp * error

        if self.prev_error is None:
            D_ctrl = 0
        else:
            D_ctrl = self.kd * (error - self.prev_error)

        self.prev_error = error

        return K_ctrl + D_ctrl