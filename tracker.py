from sortb import *
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from utils import centered_events

#Tracker class that manages all tracks
class Tracker:

    def __init__(self, max_distance =50,t_found =3,t_lost=5 ):
        self.tracks = []
        self.next_id = 0
        self.distance = max_distance
        self.t_found = t_found
        self.t_lost = t_lost

    def get_tracks(self):
        return self.tracks 
    
    def get_track(self, track_id):
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
    
    def get_all_centroids(self):

        if len(self.tracks) == 0:
            return np.empty((0, 2))

        return np.array([[ *track.get_centroid(), track.id] for track in self.tracks])
     

    def add_track(self, measurement, current_t):
        cluster, x, y, area = measurement
        track = Sortb(x, y, area, current_t, self.next_id)
        self.tracks.append(track)
        self.next_id += 1


    def hun_matching(self, measurements):
        if len(self.tracks) == 0 or len(measurements) == 0:
            return [], np.arange(len(self.tracks)), np.arange(len(measurements))
    
        track_xy = np.array([track.X[:2, 0] for track in self.tracks])
        meas_xy = measurements[:, 1:3]

        track_xy = track_xy.reshape(-1, 2)
        meas_xy = meas_xy.reshape(-1, 2)

        cost = distance_matrix(track_xy, meas_xy)
        row_ind, col_ind = linear_sum_assignment(cost)

        track_matched = np.zeros(len(self.tracks), dtype=bool)
        meas_matched  = np.zeros(len(measurements), dtype=bool)
        matched = []

        for r, c in zip(row_ind, col_ind):

            if cost[r, c] <= self.distance:

                matched.append((r, c))

                track_matched[r] = True
                meas_matched[c] = True

        unmatched_tracks = np.where(~track_matched)[0]
        unmatched_measurements = np.where(~meas_matched)[0]

        return matched, unmatched_tracks, unmatched_measurements



    
    def update_tracks(self, measurments, current_t):
        measurment_to_track_dict =[]
        if len(self.tracks) == 0:
            for measurement in measurments:
                self.add_track(measurement, current_t)
                cluster_id = int(measurement[0])
                track_id = self.tracks[-1].id
                
                measurment_to_track_dict.append([cluster_id, track_id])
            return measurment_to_track_dict
        
        for track in self.tracks:
            track.predict(current_t)
        
        matched, unmatched_tracks, unmatched_measurements = self.hun_matching(measurments)
      
        for track_idx, meas_idx in matched:
            z = measurments[meas_idx][1:4] 
            self.tracks[track_idx].update(z)
            self.tracks[track_idx].missed = 0
            self.tracks[track_idx].hits += 1
            if self.tracks[track_idx].hits >self.t_found:
                cluster_id = int(measurments[meas_idx][0])
                measurment_to_track_dict.append([cluster_id,self.tracks[track_idx].id])
            
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].missed += 1

        for meas_idx in unmatched_measurements:
            self.add_track(measurments[meas_idx], current_t)
            measurment_to_track_dict.append([measurments[meas_idx][0],self.tracks[-1].id])

        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.t_lost
            ]
        
        return measurment_to_track_dict
    
    def track_update_freq(self, track_id,events,shift):
        track = self.get_track(track_id)
        track_x,track_y = track.get_centroid()
        c_events = centered_events(events,track_x,track_y,shift = shift)
            
        max_freq = track.update_freq(c_events,size_w=shift*2,size_h=shift*2)

        return track_id, max_freq
        
       



   


