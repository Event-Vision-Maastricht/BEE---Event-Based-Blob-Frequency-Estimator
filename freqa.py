import numpy as np
    #def freqFFTcount():
    #    return 

    #def freqFFTarea():
    #    return 

    #def freqSTFFT():
    #    return 

def freqEBFM(events, width,height, min_freq=200,max_freq=300,diff_thresh_us=0,ebfm=None):
    f=np.arange(min_freq, max_freq, 1,dtype=np.complex64)
    w = 2*np.pi*f
    if ebfm is None:
        ebfm = np.zeros((width, height,len(f)),dtype=np.complex64)
    
    t_sec = events[:, 3:4] * 1e-6
    exponents = np.exp(-1j*w*t_sec).astype(np.complex64)
    for i,event in enumerate(events):
        x,y,p,t = event
        ebfm[y, x, :] += exponents[i]
    ebfm_abs = np.abs(ebfm)
    ebfm_max = np.argmax(ebfm_abs, axis=2)
    ebfm_max[ebfm_max>0] +=200
    #print(ebfm_max)

    return ebfm_max, ebfm

    

    #def freqFFTPCA(): #???
    #    return 
