import numpy as np 
import cv2
import yaml
import csv
from config import cfg

#important functions, related to calculations and visualization 
def visualizator2(events):
    frame =  np.zeros((720,1280,3), dtype=np.uint8)
    for event in events:
        x,y,polarity,timestamp, *rest = event
        if polarity ==1:
            frame[y,x,2]=255
        else:
            frame[y,x,0]=255

def visualizator3(events, cluster_labels,centroids, colours,freq):
    unique_clusters = np.unique(cluster_labels)
    colours = generate_cluster_colors( unique_clusters,colours)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x, y, polarity, timestamp, *rest = events[0]
    
    cv2.putText(frame, f"{timestamp}us", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    with open("detected_tracks.csv", "a", newline="") as f:
        writer = csv.writer(f)

        for event, label in zip(events, cluster_labels):
            if label == -1:
                continue

            x, y, polarity, timestamp, *rest = event
            writer.writerow([x, y, polarity, timestamp ,label])
        
    for event, label in zip(events, cluster_labels):
        x, y, polarity, timestamp,*rest = event
        if label == -1:
            frame[y, x] = [255, 255, 255]   
        else:
            frame[y, x] = colours[label] 
         
    if centroids.size != 0: 
        for cx, cy, id in centroids:
            cx = int(cx)
            cy = int(cy)
            max_freq = freq[id] 
            cv2.circle( frame,(cx, cy),1,(0, 255, 255),-1 )
            cv2.putText(frame, f"{max_freq}", (cx+15, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    

    return frame

def visualizator3(events, cluster_labels,centroids, colours,freq,mag):
    unique_clusters = np.unique(cluster_labels)
    colours = generate_cluster_colors( unique_clusters,colours)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x, y, polarity, timestamp, *rest = events[0]
    window_stamp = timestamp
    cv2.putText(frame, f"{window_stamp}us", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    with open(cfg["data"]["file_name"]+"_detected_tracks_t.csv", "a", newline="") as f:
        writer = csv.writer(f)

        for event, label in zip(events, cluster_labels):
            if label == -1:
                continue

            x, y, polarity, timestamp, *rest = event
            writer.writerow([x, y, polarity, timestamp ,label])
        
    for event, label in zip(events, cluster_labels):
        x, y, polarity, timestamp,*rest = event
        if label == -1:
            frame[y, x] = [255, 255, 255]   
        else:
            frame[y, x] = colours[label] 
         
    if centroids.size != 0: 
        #with open(cfg["data"]["file_name"]+"_fa.csv", "a", newline="") as f:
         with open(cfg["data"]["file_name"]+"_cluster_centroids_t.csv", "a", newline="") as f1, \
              open(cfg["data"]["file_name"]+"_f_t.csv", "a", newline="") as f2:
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            writer = csv.writer(f)
            for cx, cy, id in centroids:
                cx = int(cx)
                cy = int(cy)
                max_freq = freq[id]
                max_mag = mag[id] 
                #writer.writerow([window_stamp,id,max_freq,max_mag])
                #writer.writerow([window_stamp,id,cx,cy])
                writer1.writerow([window_stamp, id, cx, cy])
                writer2.writerow([window_stamp, id, max_freq, max_mag])
                cv2.circle( frame,(cx, cy),1,(0, 255, 255),-1 )
                cv2.putText(frame, f"{max_freq};{max_mag:.3f}", (cx+15, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    

    return frame
        
            
def get_centroid(events): 
    x = np.mean(events[:, 0])
    y = np.mean(events[:, 1])

    return x,y

import numpy as np

def get_centroid_trimmed(events, trim_percent=0.1):
    x_vals = np.sort(events[:, 0])
    y_vals = np.sort(events[:, 1])

    trim_n = int(len(events) * trim_percent)

    if trim_n > 0:
        x_vals = x_vals[trim_n:-trim_n]
        y_vals = y_vals[trim_n:-trim_n]

    x = np.mean(x_vals)
    y = np.mean(y_vals)

    return x, y

import numpy as np

def get_centroid_iqr(events, k=1.5):

    x_vals = events[:, 0]
    y_vals = events[:, 1]

    q1_x = np.percentile(x_vals, 25)
    q3_x = np.percentile(x_vals, 75)
    iqr_x = q3_x - q1_x
    x_min = q1_x - k * iqr_x
    x_max = q3_x + k * iqr_x

    q1_y = np.percentile(y_vals, 25)
    q3_y = np.percentile(y_vals, 75)
    iqr_y = q3_y - q1_y
    y_min = q1_y - k * iqr_y
    y_max = q3_y + k * iqr_y

    mask = (
        (x_vals >= x_min) & (x_vals <= x_max) &
        (y_vals >= y_min) & (y_vals <= y_max)
    )

    filtered_events = events[mask]

    if len(filtered_events) == 0:
        return get_centroid_trimmed(events)

    x = np.mean(filtered_events[:, 0])
    y = np.mean(filtered_events[:, 1])

    return x, y

def get_area(events):
    xmin = np.min(events[:, 0])
    xmax = np.max(events[:, 0])

    ymin = np.min(events[:, 1])
    ymax = np.max(events[:, 1])

    w = xmax - xmin
    h = ymax - ymin

    a= w * h 
    return a

def get_measurment(cluster): 
    #x, y = get_centroid_iqr(cluster,k=0.5)
    #x, y = get_centroid_trimmed(cluster, trim_percent=0.1)
    x, y = get_centroid(cluster)
    a = get_area(cluster)
    return x,y,a

def get_all_measurements(frame_events, clusters):
    unique_clusters = np.unique(clusters)
    measurements = []

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        cluster_events = frame_events[clusters == cluster]
        x,y,a = get_measurment(cluster_events)
        measurements.append([cluster,x,y,a])

    if len(measurements) == 0:
        return np.empty((0, 4))

    return np.array(measurements)
    
def generate_cluster_colors1(unique_clusters):
    global colours
    np.random.seed(0)
    colors = {}

    for cluster in unique_clusters:
        colors[cluster] = np.random.randint(0, 256, 3, dtype=np.uint8)
     
    return colors

def generate_cluster_colors(unique_clusters, colours):

    for cluster in unique_clusters:

        if cluster == -1:
            continue

        if cluster not in colours:
            colours[cluster] = np.random.randint(0, 256, 3, dtype=np.uint8)

    return colours

def centered_events(events, centroid_x, centroid_y, shift=60):
    events_c = events.copy()

    shift_x = centroid_x - shift
    shift_y = centroid_y - shift

    events_c[:, 0] = events_c[:, 0] - shift_x
    events_c[:, 1] = events_c[:, 1] - shift_y

    return events_c