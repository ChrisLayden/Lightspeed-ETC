# Chris Layden

'''Defining sensors and telescopes.'''

import os
import pysynphot as S
import numpy as np
from observatory import Observatory, Sensor, Telescope

data_folder = os.path.dirname(__file__) + '/../data/'

# Defining sensors
# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)

cosmos_arr = np.genfromtxt(data_folder + 'cosmos_qe_datasheet.csv', delimiter=',')
cosmos_qe = S.ArrayBandpass(cosmos_arr[:, 0] * 10, cosmos_arr[:, 1])
cosmos = Sensor(pix_size=10, read_noise=1.0, dark_current=0.005, full_well=80000,
                qe=cosmos_qe)

qcmos_arr = np.genfromtxt(data_folder + 'qCMOS_QE.csv', delimiter=',')
qcmos_qe = S.ArrayBandpass(qcmos_arr[:, 0] * 10, qcmos_arr[:, 1])
qcmos = Sensor(pix_size=4.6, read_noise=0.29, dark_current=0.006, full_well=7000,
               qe=qcmos_qe)

basic_sensor = Sensor(pix_size=10, read_noise=10, dark_current=0.01,
                      full_well=100000)

# Sensor dictionary for ground-based observatories
sensor_dict_gb = {'Define New Sensor': basic_sensor, 'Sony IMX 455': imx455,
                  'COSMOS': cosmos, 'qCMOS': qcmos}

sensor_dict_lightspeed = {'Define New Sensor': basic_sensor, 'qCMOS': qcmos}
                  
# Defining telescopes
basic_tele = Telescope(diam=10, f_num=1)

magellan_bandpass = S.UniformTransmission(0.95)
magellan_tele_native = Telescope(diam=650, f_num=11, bandpass=magellan_bandpass)
magellan_tele_lightspeed = Telescope(diam=650, f_num=1.4, bandpass=magellan_bandpass)
magellan_tele_prototype = Telescope(diam=650, f_num=2.75, bandpass=magellan_bandpass)

hale_bandpass = S.UniformTransmission(0.95)
hale_tele = Telescope(diam=510, f_num=3.29, psf_type='airy', bandpass=hale_bandpass)
winter_bandpass = S.UniformTransmission(0.23)
winter_tele = Telescope(diam=100, f_num=6.0, psf_type='airy', bandpass=winter_bandpass)

# Telescope dictionary for ground-based observatories
telescope_dict_gb = {'Define New Telescope': basic_tele, 
                     'Magellan': magellan_tele_native,
                     'Magellan LightSpeed': magellan_tele_lightspeed,
                     'Hale': hale_tele}

telescope_dict_lightspeed = {'Define New Telescope': basic_tele,
                             'Magellan Native': magellan_tele_native,
                             'Magellan Prototype': magellan_tele_prototype,
                             'Magellan Lightspeed': magellan_tele_lightspeed,
                             'Hale': hale_tele, 'WINTER': winter_tele}

# Defining filters
no_filter = S.UniformTransmission(1)
johnson_u = S.ObsBandpass('johnson,u')
johnson_b = S.ObsBandpass('johnson,b')
johnson_v = S.ObsBandpass('johnson,v')
johnson_r = S.ObsBandpass('johnson,r')
johnson_i = S.ObsBandpass('johnson,i')
johnson_j = S.ObsBandpass('johnson,j')
# Array with uniform total transmission 9000-17000 ang
swir_wave = np.arange(9000, 17000, 100)
swir_thru = np.ones(len(swir_wave))
swir_filt_arr = np.array([swir_wave, swir_thru]).T
# Pad with zeros at 8900 and 17100 ang
swir_filt_arr = np.vstack(([8900, 0], swir_filt_arr, [17100, 0]))
swir_filter = S.ArrayBandpass(swir_filt_arr[:, 0], swir_filt_arr[:, 1])

sloan_uprime = np.genfromtxt(data_folder + 'SLOAN_SDSS.uprime_filter.dat',
                             delimiter='\t')
sloan_uprime = S.ArrayBandpass(sloan_uprime[:, 0], sloan_uprime[:, 1])
sloan_gprime = np.genfromtxt(data_folder + 'SLOAN_SDSS.gprime_filter.dat',
                             delimiter='\t')
sloan_gprime = S.ArrayBandpass(sloan_gprime[:, 0], sloan_gprime[:, 1])
sloan_rprime = np.genfromtxt(data_folder + 'SLOAN_SDSS.rprime_filter.dat',
                             delimiter='\t')
sloan_rprime = S.ArrayBandpass(sloan_rprime[:, 0], sloan_rprime[:, 1])
sloan_iprime = np.genfromtxt(data_folder + 'SLOAN_SDSS.iprime_filter.dat',
                             delimiter='\t')
sloan_iprime = S.ArrayBandpass(sloan_iprime[:, 0], sloan_iprime[:, 1])
sloan_zprime = np.genfromtxt(data_folder + 'SLOAN_SDSS.zprime_filter.dat',
                             delimiter='\t')
sloan_zprime = S.ArrayBandpass(sloan_zprime[:, 0], sloan_zprime[:, 1])


# Array with uniform total transmission 4000-7000 ang
vis_wave = np.arange(4000, 7000, 100)
vis_thru = np.ones(len(vis_wave))
vis_filt_arr = np.array([vis_wave, vis_thru]).T
# Pad with zeros
vis_filt_arr = np.vstack(([3900, 0], vis_filt_arr, [7100, 0]))
vis_filter = S.ArrayBandpass(vis_filt_arr[:, 0], vis_filt_arr[:, 1])

filter_dict_gb = {'None': no_filter, 'Johnson U': johnson_u,
                  'Johnson B': johnson_b, 'Johnson V': johnson_v,
                  'Johnson R': johnson_r, 'Johnson I': johnson_i,
                  'Johnson J': johnson_j, 
                  'Sloan Uprime': sloan_uprime, 'Sloan Gprime': sloan_gprime, 'Sloan Rprime': sloan_rprime,
                  'SWIR (900-1700 nm 100%)': swir_filter,
                  'Visible (400-700 nm 100%)': vis_filter}

filter_dict_lightspeed = {"None": no_filter, "Sloan u'": sloan_uprime,
                          "Sloan g'": sloan_gprime, "Sloan r'": sloan_rprime,
                          "Sloan i'": sloan_iprime, "Sloan z'": sloan_zprime}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.plot(sloan_uprime.wave, sloan_uprime.throughput, label='Sloan u\'')
    plt.plot(sloan_gprime.wave, sloan_gprime.throughput, label='Sloan g\'')
    plt.plot(sloan_rprime.wave, sloan_rprime.throughput, label='Sloan r\'')
    plt.plot(sloan_iprime.wave, sloan_iprime.throughput, label='Sloan i\'')
    plt.plot(sloan_zprime.wave, sloan_zprime.throughput, label='Sloan z\'')
    plt.legend()
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Throughput')
    plt.show()
