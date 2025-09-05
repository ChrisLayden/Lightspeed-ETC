# Chris Layden

'''Defining sensors and telescopes.'''

import os
import pysynphot as S
import numpy as np
from observatory import Sensor, Telescope

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
qcmos = Sensor(pix_size=4.6, read_noise=0.29, dark_current=0.004, full_well=7000,
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

# Halpha filter
ha_filter = np.genfromtxt(data_folder + 'halpha_filter.csv', delimiter=',', skip_header=1)
ha_filter = S.ArrayBandpass(ha_filter[:, 0] * 10, ha_filter[:, 1] / 100)

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
                  'Sloan Uprime': sloan_uprime, 'Sloan Gprime': sloan_gprime,
                  'Sloan Rprime': sloan_rprime,
                  'SWIR (900-1700 nm 100%)': swir_filter,
                  'Visible (400-700 nm 100%)': vis_filter}

filter_dict_lightspeed = {"None": no_filter, "Sloan u'": sloan_uprime,
                          "Sloan g'": sloan_gprime, "Sloan r'": sloan_rprime,
                          "Sloan i'": sloan_iprime, "Sloan z'": sloan_zprime}

# Bandpass representing transmission through the atmosphere at airmass 1
atmo_bandpass = np.genfromtxt(data_folder + 'atmo_transmission_airmass1.csv',
                              delimiter=',')
atmo_bandpass = S.ArrayBandpass(atmo_bandpass[:, 0] * 10, atmo_bandpass[:, 1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Transmission plots for SDSS filters and for QE
    xpoints_oiii = np.arange(485, 515, 0.5)
    ypoints_oiii = 0.97 * np.exp(-0.5 * ((xpoints_oiii - 500.7) / 3.822)**2)
    plt.rcParams.update({'font.size': 14})
    # plt.figure(figsize=(12, 6))
    # plt.plot(sloan_uprime.wave / 10, sloan_uprime.throughput * 100, 'b--', label='SDSS u\'', alpha=0.5)
    # plt.plot(sloan_gprime.wave / 10, sloan_gprime.throughput * 100, 'g--', label='SDSS g\'', alpha=0.5)
    # plt.plot(sloan_rprime.wave / 10, sloan_rprime.throughput * 100, 'r--', label='SDSS r\'', alpha=0.5)
    # plt.plot(sloan_iprime.wave / 10, sloan_iprime.throughput * 100, 'm--', label='SDSS i\'', alpha=0.5)
    # plt.plot(sloan_zprime.wave / 10, sloan_zprime.throughput * 100, 'c--', label='SDSS z\'', alpha=0.5)
    # plt.plot(xpoints_oiii, ypoints_oiii * 100, '-.', color='darkorange', label='Baader OIII', alpha=0.5)
    # plt.plot(ha_filter.wave / 10, ha_filter.throughput * 100, '-.', color='darkred', label=r'Alluxa $H_\alpha$', alpha=0.5)
    # # plt.plot(atmo_bandpass.wave / 10, atmo_bandpass.throughput * 100, 'k:', label='Atmosphere', alpha=0.5)
    # plt.plot(qcmos_qe.wave / 10, qcmos_qe.throughput * 100, 'k', label='Datasheet QE', alpha=0.5)
    # qCMOS_meas_QE_wavelengths = np.array([296.7, 400., 500., 550., 600., 640., 700., 800., 900., 1000., 1064.])
    # qCMOS_meas_QE = np.array([0.3756, 0.8271, 0.8604, 0.8193, 0.7585, 0.6932, 0.6111, 0.4982, 0.3117, 0.0932, 0.0115])
    # xerr = 5
    # yerr = 0.05 * 100
    # plt.errorbar(qCMOS_meas_QE_wavelengths, qCMOS_meas_QE * 100, xerr=xerr, yerr=yerr, label='Measured QE', fmt='k.', markersize=1)
    # # Put legend above the plot
    # plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.76, 0.65), fontsize=12)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Transmission (%)')
    # plt.xlim(250, 1100)
    # plt.ylim(0,100)
    # plt.show()
    plt.figure(figsize=(10, 6))
    tot_thru_u = sloan_uprime * qcmos_qe * atmo_bandpass * 0.05
    tot_thru_g = sloan_gprime * qcmos_qe * atmo_bandpass * 0.57
    tot_thru_r = sloan_rprime * qcmos_qe * atmo_bandpass * 0.65
    tot_thru_i = sloan_iprime * qcmos_qe * atmo_bandpass * 0.28
    tot_thru_z = sloan_zprime * qcmos_qe * atmo_bandpass * 0.06
    tot_thru_oiii = S.ArrayBandpass(xpoints_oiii * 10, ypoints_oiii) * qcmos_qe * atmo_bandpass * 0.57
    tot_thru_halpha = ha_filter * qcmos_qe * atmo_bandpass * 0.65
    plt.plot(tot_thru_u.wave / 10, tot_thru_u.throughput * 100, 'b--', label='SDSS u\'', alpha=0.5)
    plt.plot(tot_thru_g.wave / 10, tot_thru_g.throughput * 100, 'g--', label='SDSS g\'', alpha=0.5)
    plt.plot(tot_thru_r.wave / 10, tot_thru_r.throughput * 100, 'r--', label='SDSS r\'', alpha=0.5)
    plt.plot(tot_thru_i.wave / 10, tot_thru_i.throughput * 100, 'm--', label='SDSS i\'', alpha=0.5)
    plt.plot(tot_thru_z.wave / 10, tot_thru_z.throughput * 100, 'c--', label='SDSS z\'', alpha=0.5)
    plt.plot(tot_thru_oiii.wave / 10, tot_thru_oiii.throughput * 100, '-.', color='darkorange', label='Baader OIII', alpha=0.5)
    plt.plot(tot_thru_halpha.wave / 10, tot_thru_halpha.throughput * 100, '-.', color='darkred', label='Halpha', alpha=0.5)
    # Use text to label the 5 curves
    plt.text(tot_thru_u.pivot() / 10, np.max(tot_thru_u.throughput) * 100 + 0.5, "u'", color='b', ha='center')
    plt.text(tot_thru_g.pivot() / 10, np.max(tot_thru_g.throughput) * 100 + 0.5, "g'", color='g', ha='center')
    plt.text(tot_thru_r.pivot() / 10, np.max(tot_thru_r.throughput) * 100 + 0.5, "r'", color='r', ha='center')
    plt.text(tot_thru_i.pivot() / 10, np.max(tot_thru_i.throughput) * 100 + 0.5, "i'", color='m', ha='center')
    plt.text(tot_thru_z.pivot() / 10, np.max(tot_thru_z.throughput) * 100 + 0.5, "z'", color='c', ha='center')
    plt.text(500.7, np.max(tot_thru_oiii.throughput) * 100 + 0.5, "OIII", color='darkorange', ha='center')
    plt.text(656.3, np.max(tot_thru_halpha.throughput) * 100 + 0.5, r"$H_\alpha$", color='darkred', ha='center')
    plt.xlim(300, 1050)
    plt.ylim(0.1, 40)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.show()

