import numpy as np
#frequency algorithms

    #def freqFFTcount():
    #    return 

    #def freqFFTarea():
    #    return 

    #def freqSTFFT():
    #    return 

def freqEBFM(events, width,height, min_freq=200,max_freq=300,freq_res= 1,ebfm=None):
    f=np.arange(min_freq, max_freq,freq_res,dtype=np.float32)
    w = 2*np.pi*f
    if ebfm is None:
        ebfm = np.zeros((height, width,len(f)),dtype=np.complex64)
    
    t_sec = events[:, 3:4] * 1e-6
    exponents = np.exp(-1j*w*t_sec).astype(np.complex64)
    for i,event in enumerate(events):
        x,y,p,t = event
        try:
            ebfm[y, x, :] += exponents[i]
        except: 
            #print("event out of bounds of box_shift, needs a higher value",flush=True)
            continue
    ebfm_abs = np.abs(ebfm)
    ebfm_max = np.argmax(ebfm_abs, axis=2)
    flat_idx = np.argmax(ebfm_abs)
    y_max, x_max, f_idx = np.unravel_index(flat_idx, ebfm_abs.shape)
    ebfm_max +=min_freq
    cur_max_freq= ebfm_max[y_max,x_max]


    return cur_max_freq, ebfm
    
    
    #def freqFFTPCA(): #???
    #    return 
