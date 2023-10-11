
import garpen_runner

import h5py
import hastypy.util.plot_utility as pu
import numpy as np
import scipy as sp
import scipy.signal as spsig

import matplotlib.pyplot as plt

base_path = 'D:/4Drecon/Garpen/Ena/'

from nibabel.viewers import OrthoSlicer3D



#def nifti_plot(image):
#	
#	OrthoSlicer3D(image[0,...])


def resp_gating(gating):

    tvec = 0

    timegate = gating['TIME_E0'][()][0,:]
    respgate = gating['RESP_E0'][()][0,:]


    timeidx = np.argsort(timegate)
    
    time = timegate[timeidx]
    resp = respgate[timeidx]

    Fs = 1.0 / (time[1] - time[0])
    Fn = Fs / 2.0
    Fco = 0.01
    Fsb = 1.0
    Rp = 0.05
    Rs = 50.0

    def torad(x):
        return x

    N, Wn = spsig.buttord(torad(Fco / Fn), torad(Fsb / Fn), Rp, Rs, analog=True)
    print(Wn)
    b, a = spsig.butter(N, Wn)
    filtered = spsig.filtfilt(b, a, resp)
    
    hfilt = spsig.hilbert(resp - filtered)



    plt.figure()
    plt.plot(resp)
    plt.plot(filtered)
    plt.show()

    plt.figure()
    plt.plot(resp - filtered)
    plt.show()

    hfilt = spsig.hilbert(resp - filtered)

    plt.figure()
    plt.plot((resp - filtered) / np.max(resp - filtered))
    plt.plot(np.angle(hfilt))
    plt.show()


with h5py.File('D:\\4DRecon\\Wahlin\\BrutalAndning\\MRI_Raw.h5') as f:
    gating = f['Gating']

    resp_gating(gating)
    

plot_spoke_info = False
if plot_spoke_info:
    with h5py.File(base_path + 'MRI_Raw.h5', "r") as f:
        kdata = f['Kdata']

        def plotter(spoke_number):

            kx = kdata['KX_E0'][()]
            ky = kdata['KY_E0'][()]
            kz = kdata['KZ_E0'][()]
            
            kw = kdata['KW_E0'][()]

            xspoke = kx[0,spoke_number,:]
            yspoke = ky[0,spoke_number,:]
            zspoke = kz[0,spoke_number,:]

            wspoke = kw[0,spoke_number,:]
            plt.figure()
            plt.plot(wspoke)
            plt.title('Weights')
            plt.show()

            pu.scatter_3d(np.stack([xspoke, yspoke, zspoke]))

            r = np.sqrt(xspoke**2 + yspoke**2 + zspoke**2)

            plt.figure()
            plt.plot(r)
            plt.title('Radius')
            plt.show()

            plt.figure()
            plt.plot(xspoke, 'r-*')
            plt.plot(yspoke, 'g-*')
            plt.plot(zspoke, 'b-*')
            plt.title('Coords')
            plt.legend(['X', 'Y', 'Z'])
            plt.show()


        def resp_gating():
            gating = f['Gating']

            tvec = 0

            timegate = gating['TIME_E0'][()][0,:]
            respgate = gating['RESP_E0'][()][0,:]


            timeidx = np.argsort(timegate)
            
            time = timegate[timeidx]

            resp = respgate[timeidx]

            Fs = 0.2 * 1.0 / (time[1] - time[0])
            Fn = Fs / 2.0
            Fco = 0.01
            Fsb = 10.0
            Rp = 1.0
            Rs = 50.0

            N, Wn = spsig.buttord(Fco / Fn, Fsb / Fn, Rp, Rs)
            b, a = spsig.butter(N, Wn)
            sos = spsig.tf2sos(b,a)
            filtered = spsig.sosfilt(sos, resp)

            plt.figure()
            plt.plot(resp)
            plt.plot(filtered)
            plt.show()

            plt.figure()
            plt.plot(resp - filtered)
            plt.show()

            hfilt = spsig.hilbert(resp - filtered)

            plt.figure()
            plt.plot((resp - filtered) / np.max(resp - filtered))
            plt.plot(np.angle(hfilt))
            plt.show()

        plotter(0)