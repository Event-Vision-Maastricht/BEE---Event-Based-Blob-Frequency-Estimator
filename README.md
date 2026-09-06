# BEE - Event-Based Blob Frequency Estimator 
## GitHub of *BEE: Blob Frequency Estimation Of Fast Moving High-Frequency Objects Using Event-Based Data* 2026 paper 


This projects implements event-based blob tracking using DBSCAN 
for detection and the SORT algorithm(Kalman filter + Hungarian matching) for tracking[1]. As well as, frequecny estimation using Event-Based Frequency Mapping[2].  



## Dependancies:
- pandas
- opencv-python
- scikit-learn
- scipy
- PyYAML
- coded on Python 3.12.3

### Configs
- congig_bees.yaml fine tuned for Bee Swarm Dataset[3] 
- config_mbs.yaml fine tuned for BeeVent
- config_inscombc.yaml for individual flight paths of Insect Combined dataset[4]

### References  

[1] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, *Simple online and realtime tracking,* arXiv (Cornell University), Feb. 2016, doi: 10.48550/arxiv.1602.00763.

[2] Aitsam, M., Goyal, G., Bartolozzi, C., & Di Nuovo, A. (2024). *Vibration Vision: Real-Time Machinery Fault Diagnosis with Event Cameras*. In *Proceedings of ECCV-NeVi Workshop*.

[3] A. Apps, Z. Wang, V. Perejogin, T. Molloy, and R. Mahony, *Asynchronous Multi-Object Tracking with an Event Camera,* arXiv (Cornell University), May 2025, doi: 10.48550/arxiv.2505.08126.

[4]Pohle-Fröhlich, R., Gebler, C., Böge, M., Bolten, T., Gehlen, L., Glück, M., Traynor, K. *Features for Classifying Insect Trajectories in Event Camera Recordings.* pp. 355–364 (Jan 2025). https://doi.org/10.5220/0013140100003912