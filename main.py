import pandas as pd
import cv2
from sklearn.cluster import DBSCAN
from utils import *
from tracker import *
from config import cfg

#event processing
csv_path = cfg["data"]["csv_path"]
#change this for your csv if needed, currently skips first row to handle both csvs with header and no header, but skips first events if there is no header 
events_df = pd.read_csv(csv_path, skiprows=1, header=None, names=['x','y','p','t'])
events = events_df[['x','y','p','t']].values
H, W = cfg["data"]["H"], cfg["data"]["W"]

initial_len =  cfg["data"]["initial_len"]
window_len = cfg["data"]["window_len"]

#
eps = cfg["dbscan"]["eps"]
min_samples = cfg["dbscan"]["min_samples"]
time_scale = cfg["dbscan"]["time_scale"]
time_stamps = events[:, 3]
#first event time 
t_first = events[0, 3]
#last event time
t_last = events[-1, 3]
#initial window
t0 = t_first
t1 = t_first+ initial_len
paused = False
colours = {}
freq={}
tracker = Tracker(max_distance =cfg["kalman"]["sort"]["max_distance"], t_found=cfg["kalman"]["sort"]["t_found"],t_lost=cfg["kalman"]["sort"]["t_lost"])
#---
#max_distance =cfg["kalman"]["sort"]["max_distance"]
save_video = cfg["save_video_p"]["save_video"]
if save_video:
    fps = cfg["save_video_p"]["fps"]
    out_path = cfg["save_video_p"]["out_path"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

#-----------main loop processing events 
while t0 < t_last:
        # get og events 
        id_f = np.searchsorted(time_stamps, t0, side='left')
        id_s = np.searchsorted(time_stamps, t1, side='right')  
        frame_events = events[id_f:id_s] 
        if len(frame_events) == 0:
            t0 += window_len
            t1 += window_len
            continue
        #now 3D DBSCAN 
        
        #-------------------Detector
        #get x,y,timestamp(ignoring polarity for now)
        X3D = np.column_stack((frame_events[:, 0],frame_events[:, 1], frame_events[:, 3]*time_scale))
        #dbscan
        db = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = db.fit_predict(X3D)
        #gather unique labels 
        #-------------------Detector
        #-----------Measurments
        measurements = get_all_measurements(frame_events, clusters)
        #-----------Measurments
        #-----------Kalman
        measurment_to_track_dict = tracker.update_tracks(measurements,t0) 
        #a dictionary 
        #-----------Kalman
        #-----------Tracks 
        persistent_labels = np.full_like(clusters, -1)
        
        for cluster_id, track_id in measurment_to_track_dict:
            #persistance of tracks trough random labeles from DBSCAN
            persistent_labels[clusters == cluster_id] = track_id
            #frequency
        #-----------Tracks
        unique_c = np.unique(persistent_labels)
        #-----------Update frequency
        for cluster  in unique_c:
            if cluster  == -1:
                continue
            c_events = frame_events[persistent_labels == cluster]
            track_id, max_freq =tracker.track_update_freq(cluster,c_events,cfg["frequency"]["box_shift"])
            #frequency 
            freq[track_id]= max_freq
        #-----------Update frequency

        centroids = tracker.get_all_centroids()
        
        
        #------------intuatie visualization
        event_frame_dt = visualizator3(frame_events,persistent_labels,centroids,colours,freq)
        #save video if needed 
        if save_video:
            video_writer.write(event_frame_dt)
        #event_frame_dt = cv2.resize(event_frame_dt, (640, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Event Stream", event_frame_dt)
        key = cv2.waitKey(100)
        if key == 27:  
            break
        elif key == ord('p'):  
            paused = not paused

        while paused:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('p'):  
                paused = False
            elif key2 == 27:
                exit()
       
        t0 += window_len
        t1 += window_len
        #------------intuitive visualization
#---------end of main loop processing events 
if save_video:
    video_writer.release()
cv2.destroyAllWindows()
