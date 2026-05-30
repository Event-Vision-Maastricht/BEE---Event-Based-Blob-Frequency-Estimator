import numpy as np 
from config import cfg
from freqa import freqEBFM

#kalman filter/object version of a track
class Sortb: 

    def update_F(self, dt):
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=float)

    def __init__(self,x,y,a,t,id):
        self.id = id
        self.t_last = t
        self.missed = 0
        self.hits = 1
        self.ebfm = None
        self.max_freq=0
        #state
        self.X = np.array([
            [x],
            [y],
            [a],
            [0],
            [0],
            [0]
        ], dtype=float)
        #transition matrix
        self.update_F(1)
        #measurment matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)
        #initial uncertainity, evolves with track
        self.P = np.diag([cfg["kalman"]["initial_uncertainty_P"]["x"], 
                          cfg["kalman"]["initial_uncertainty_P"]["y"], 
                          cfg["kalman"]["initial_uncertainty_P"]["area"], 
                          cfg["kalman"]["initial_uncertainty_P"]["vx"], 
                          cfg["kalman"]["initial_uncertainty_P"]["vy"], 
                          cfg["kalman"]["initial_uncertainty_P"]["varea"]]).astype(float)
        #process noise
        cfg["kalman"]["measurement_noise_R"]["x"]
        self.Q = np.diag([cfg["kalman"]["process_noise_Q"]["x"], 
                          cfg["kalman"]["process_noise_Q"]["y"], 
                          cfg["kalman"]["process_noise_Q"]["area"],
                          cfg["kalman"]["process_noise_Q"]["vx"], 
                          cfg["kalman"]["process_noise_Q"]["vy"],
                          cfg["kalman"]["process_noise_Q"]["varea"]]).astype(float)
        #measurment noise
        self.R = np.diag([cfg["kalman"]["measurement_noise_R"]["x"],
                          cfg["kalman"]["measurement_noise_R"]["y"], 
                          cfg["kalman"]["measurement_noise_R"]["area"]]).astype(float)
 
    def predict(self, current_t):
        dt = (current_t - self.t_last) * 1e-6  # use if timestamps are microseconds
        self.t_last = current_t
        #update transition matrix
        self.update_F(dt)
        # x_= F*x
        self.X = self.F @ self.X
        #P = F*P*Ft+Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        
    def update(self , z):
        z = np.asarray(z, dtype=float).reshape(3, 1)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X  = self.X + self.K @ (z-self.H @ self.X)
        self.P = self.P - self.K @ self.H @ self.P
    
    def get_state(self):
        return self.X.flatten()

    def get_centroid(self):
        return self.X[0, 0], self.X[1, 0]

    def get_area(self):
        return self.X[2, 0]
    
    def update_freq(self, events,size_w,size_h):
        self.max_freq, self.ebfm = freqEBFM(events, size_w,size_h, min_freq=cfg["frequency"]["min_freq"],max_freq=cfg["frequency"]["max_freq"],freq_res= cfg["frequency"]["freq_res"],ebfm=self.ebfm)
        return self.max_freq

    


