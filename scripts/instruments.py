'''Defining sensors and telescopes.'''

import os
from synphot import SpectralElement, ConstFlux1D, Empirical1D
import astropy.units as u
import numpy as np
from observatory import Sensor, Telescope
import matplotlib.pyplot as plt

data_folder = os.path.dirname(__file__) + '/../data/'

# Defining sensors
# Sensor dark currents given at -25 deg C
imx455_qe = SpectralElement.from_file(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)
# Have not implemented severe nonlinearity scaling for COSMOS
cosmos_arr = np.genfromtxt(data_folder + 'cosmos_qe_datasheet.csv', delimiter=',')
cosmos_qe = SpectralElement(Empirical1D, points=cosmos_arr[:, 0] * 10 * u.AA,
                          lookup_table=cosmos_arr[:, 1])
cosmos = Sensor(pix_size=10, read_noise=1.0, dark_current=0.005, full_well=80000,
                qe=cosmos_qe)

qcmos_arr = np.genfromtxt(data_folder + 'qCMOS_QE.csv', delimiter=',')
qcmos_qe = SpectralElement(Empirical1D, points=qcmos_arr[:, 0] * 10 * u.AA,
                         lookup_table=qcmos_arr[:, 1])
qcmos_nonlinearity = np.genfromtxt(data_folder + 'qCMOS_nonlinearity_scaling_2.csv', delimiter=',', skip_header=1)
qcmos = Sensor(pix_size=4.6, read_noise=0.28, dark_current=0.007, full_well=7000,
               qe=qcmos_qe, nonlinearity_scaleup=qcmos_nonlinearity)

noiseless_ideal_sensor = Sensor(pix_size=4.6, read_noise=0.0, dark_current=0.004, full_well=7000,
                                qe=qcmos_qe, nonlinearity_scaleup=None)

basic_sensor = Sensor(pix_size=10, read_noise=10, dark_current=0.01,
                      full_well=100000, qe=SpectralElement(ConstFlux1D, amplitude=0.5))

hipercam_sensor = Sensor(pix_size=15, read_noise=4.5, dark_current=0.03, full_well=120000,
                         qe=SpectralElement(ConstFlux1D, amplitude=1.0))

# Sensor dictionary for ground-based observatories
sensor_dict_gb = {'Define New Sensor': basic_sensor, 'Sony IMX 455': imx455,
                  'COSMOS': cosmos, 'qCMOS': qcmos}

sensor_dict_lightspeed = {'Define New Sensor': basic_sensor, 'qCMOS': qcmos, 'HiPERCAM': hipercam_sensor,
                          'Noiseless Ideal Sensor': noiseless_ideal_sensor}

# Defining telescopes
basic_tele = Telescope(diam=10, f_num=1)

magellan_obscuration = 1 - 0.29 ** 2  # Diameter ratio 0.29
swope_obscuration = 1 - 0.5 ** 2  # Diameter ratio 0.5
magellan_nase_thru = 0.9 * 0.9 * 0.9 # Assuming 90% reflectivity for each of 3 mirrors
Clay_bandpass = SpectralElement(ConstFlux1D, amplitude=magellan_obscuration * magellan_nase_thru)
Clay_prime_bandpass = SpectralElement(ConstFlux1D, amplitude=magellan_obscuration * 0.9)
Clay_tele_native = Telescope(diam=650, f_num=11, bandpass=Clay_bandpass)
Clay_tele_lightspeed = Telescope(diam=650, f_num=1.4, bandpass=Clay_bandpass)
Clay_tele_prototype = Telescope(diam=650, f_num=2.9, bandpass=Clay_bandpass)
Clay_tele_prime = Telescope(diam=650, f_num=1.25, bandpass=Clay_prime_bandpass)

gtc_bandpass = SpectralElement(ConstFlux1D, amplitude=magellan_obscuration * magellan_nase_thru)
gtc_hipercam_tele = Telescope(diam=1040, f_num=2.64, bandpass=gtc_bandpass)

gmt_tele = Telescope(diam=2540, f_num=1.2, bandpass=magellan_obscuration * magellan_nase_thru)

hale_bandpass = SpectralElement(ConstFlux1D, amplitude=0.9)
hale_tele = Telescope(diam=510, f_num=3.29, psf_type='airy', bandpass=hale_bandpass)
winter_bandpass = SpectralElement(ConstFlux1D, amplitude=0.23)
winter_tele = Telescope(diam=100, f_num=6.0, psf_type='airy', bandpass=winter_bandpass)

keck_tele_zimager = Telescope(diam=1000, f_num=1.4, bandpass=magellan_obscuration * magellan_nase_thru)

swope = Telescope(diam=100, f_num=7.0, bandpass=magellan_nase_thru * swope_obscuration)

# Telescope dictionary for ground-based observatories
telescope_dict_gb = {'Define New Telescope': basic_tele,
                     'Clay': Clay_tele_native,
                     'Clay LightSpeed': Clay_tele_lightspeed,
                     'Clay Prime Focus': Clay_tele_prime,
                     'Hale': hale_tele}

telescope_dict_lightspeed = {'Define New Telescope': basic_tele,
                             'Clay Native': Clay_tele_native,
                             'Clay (proto-Lightspeed)': Clay_tele_prototype,
                             'Clay (full Lightspeed)': Clay_tele_lightspeed,
                             'Clay Prime Focus': Clay_tele_prime,
                             'Hale': hale_tele,
                             'GTC (HiPERCAM)': gtc_hipercam_tele,
                             'GMT': gmt_tele, 'Keck (Z-imager)': keck_tele_zimager,
                             'Swope': swope}

# Defining filters
no_filter = SpectralElement(ConstFlux1D, amplitude=1.0)
# Johnson filters will be downloaded by synphot if not already.
# Currently unused for Lightspeed, so commented out to prevent
# unnecessary downloads.
# johnson_u = SpectralElement.from_filter('johnson_u')
# johnson_b = SpectralElement.from_filter('johnson_b')
# johnson_v = SpectralElement.from_filter('johnson_v')
# johnson_r = SpectralElement.from_filter('johnson_r')
# johnson_i = SpectralElement.from_filter('johnson_i')
# johnson_j = SpectralElement.from_filter('johnson_j')
# Array with uniform total transmission 9000-17000 ang
swir_wave = np.arange(9000, 17000, 100)
swir_thru = np.ones(len(swir_wave))
swir_filt_arr = np.array([swir_wave, swir_thru]).T
# Pad with zeros at 8900 and 17100 ang
swir_filt_arr = np.vstack(([8900, 0], swir_filt_arr, [17100, 0]))
swir_filter = SpectralElement(Empirical1D, 
                            points=swir_filt_arr[:, 0] * u.AA,
                            lookup_table=swir_filt_arr[:, 1])

baader_uprime_data = np.genfromtxt(data_folder + 'baader_uprime_od.csv', delimiter=',', skip_header=2)
baader_uprime = SpectralElement(Empirical1D, 
                             points=baader_uprime_data[:, 0] * u.nm,
                             lookup_table= 10 ** - baader_uprime_data[:, 1])
baader_gprime_data = np.genfromtxt(data_folder + 'baader_gprime_od.csv', delimiter=',', skip_header=2)
baader_gprime = SpectralElement(Empirical1D, 
                             points=baader_gprime_data[:, 0] * u.nm,
                             lookup_table= 10 ** - baader_gprime_data[:, 1])
baader_rprime_data = np.genfromtxt(data_folder + 'baader_rprime_od.csv', delimiter=',', skip_header=2)
baader_rprime = SpectralElement(Empirical1D, 
                             points=baader_rprime_data[:, 0] * u.nm,
                             lookup_table= 10 ** - baader_rprime_data[:, 1])
baader_iprime_data = np.genfromtxt(data_folder + 'baader_iprime_od.csv', delimiter=',', skip_header=2)
baader_iprime = SpectralElement(Empirical1D, 
                             points=baader_iprime_data[:, 0] * u.nm,
                             lookup_table= 10 ** - baader_iprime_data[:, 1])
baader_zprime_data = np.genfromtxt(data_folder + 'baader_zprime_od.csv', delimiter=',', skip_header=2)
baader_zprime = SpectralElement(Empirical1D, 
                             points=baader_zprime_data[:, 0] * u.nm,
                             lookup_table= 10 ** - baader_zprime_data[:, 1])

sloan_uprime_data = np.genfromtxt(data_folder + 'SLOAN_SDSS.uprime_filter.dat',
                                delimiter='\t')
sloan_uprime = SpectralElement(Empirical1D, 
                             points=sloan_uprime_data[:, 0] * u.AA,
                             lookup_table=sloan_uprime_data[:, 1])
sloan_gprime_data = np.genfromtxt(data_folder + 'SLOAN_SDSS.gprime_filter.dat',
                                delimiter='\t')
sloan_gprime = SpectralElement(Empirical1D,
                             points=sloan_gprime_data[:, 0] * u.AA,
                             lookup_table=sloan_gprime_data[:, 1])
sloan_rprime_data = np.genfromtxt(data_folder + 'SLOAN_SDSS.rprime_filter.dat',
                                delimiter='\t')
sloan_rprime = SpectralElement(Empirical1D,
                             points=sloan_rprime_data[:, 0] * u.AA,
                             lookup_table=sloan_rprime_data[:, 1])
sloan_iprime_data = np.genfromtxt(data_folder + 'SLOAN_SDSS.iprime_filter.dat',
                                delimiter='\t')
sloan_iprime = SpectralElement(Empirical1D,
                             points=sloan_iprime_data[:, 0] * u.AA,
                             lookup_table=sloan_iprime_data[:, 1])
sloan_zprime_data = np.genfromtxt(data_folder + 'SLOAN_SDSS.zprime_filter.dat',
                                delimiter='\t')
sloan_zprime = SpectralElement(Empirical1D,
                             points=sloan_zprime_data[:, 0] * u.AA,
                             lookup_table=sloan_zprime_data[:, 1])

hipercam_gprime_data = np.genfromtxt(data_folder + 'hipercam_g.csv',
                                     delimiter=',')
hipercam_gprime = SpectralElement(Empirical1D,
                                  points=hipercam_gprime_data[:, 0] * u.nm,
                                  lookup_table=hipercam_gprime_data[:, 1])

# Halpha filter
ha_filter_data = np.genfromtxt(data_folder + 'halpha_filter.csv', delimiter=',', skip_header=1)
ha_filter = SpectralElement(Empirical1D,
                          points=ha_filter_data[:, 0] * 10 * u.AA,
                          lookup_table=ha_filter_data[:, 1] / 100)

# Array with uniform total transmission 4000-7000 ang
vis_wave = np.arange(4000, 7000, 100)
vis_thru = np.ones(len(vis_wave))
vis_filt_arr = np.array([vis_wave, vis_thru]).T
# Pad with zeros
vis_filt_arr = np.vstack(([3900, 0], vis_filt_arr, [7100, 0]))
vis_filter = SpectralElement(Empirical1D,
                           points=vis_filt_arr[:, 0] * u.AA,
                           lookup_table=vis_filt_arr[:, 1])

xpoints_oiii = np.arange(485, 515, 0.5)
ypoints_oiii = 0.97 * np.exp(-0.5 * ((xpoints_oiii - 500.7) / 3.822)**2)
baader_filter = SpectralElement(Empirical1D, 
                             points=xpoints_oiii * 10 * u.AA,
                             lookup_table=ypoints_oiii)

whitelight_throughput_proto = np.genfromtxt(data_folder + 'whitelight_throughput_proto.csv', delimiter=',')
# Account for 10% loss from fold mirror
whitelight_throughput_proto_filter = SpectralElement(Empirical1D,
                                                  points=whitelight_throughput_proto[:, 0] * u.AA,
                                                  lookup_table=whitelight_throughput_proto[:, 1] * 0.9)

# filter_dict_gb = {'None': no_filter, 'Johnson U': johnson_u,
#                   'Johnson B': johnson_b, 'Johnson V': johnson_v,
#                   'Johnson R': johnson_r, 'Johnson I': johnson_i,
#                   'Johnson J': johnson_j, 
#                   'Sloan Uprime': sloan_uprime, 'Sloan Gprime': sloan_gprime,
#                   'Sloan Rprime': sloan_rprime,
#                   'SWIR (900-1700 nm 100%)': swir_filter,
#                   'Visible (400-700 nm 100%)': vis_filter}

filter_dict_lightspeed = {"None": no_filter, "Baader u'": baader_uprime,
                          "Baader g'": baader_gprime, "Baader r'": baader_rprime,
                          "Baader i'": baader_iprime, "Baader z'": baader_zprime,
                          "Baader OIII": baader_filter, "Halpha": ha_filter,
                          "HiPERCAM g'": hipercam_gprime, "Prototype White Light": whitelight_throughput_proto_filter}

# Bandpass representing transmission through the atmosphere at airmass 1
atmo_bandpass_data = np.genfromtxt(data_folder + 'atmo_transmission_airmass1.csv',
                                 delimiter=',')
atmo_bandpass = SpectralElement(Empirical1D,
                              points=atmo_bandpass_data[:, 0] * 10 * u.AA,
                              lookup_table=atmo_bandpass_data[:, 1])

lightspeed_thru_file = os.path.join(data_folder, 'lightspeed_throughput.csv')
lightspeed_thru_data = np.genfromtxt(lightspeed_thru_file, delimiter=',', skip_header=1)
lightspeed_thru = SpectralElement(Empirical1D,
                                points=lightspeed_thru_data[:, 0] * u.um,
                                lookup_table=lightspeed_thru_data[:, -1])


if __name__ == '__main__':
    baader_uprime_file = data_folder + 'Baader-Sloan-SDSS_u-Filter.xlsx'
    # Read Excel spreadsheet into numpy array
    import pandas as pd
    # Transmission plots for SDSS filters and for QE
    xpoints_oiii = np.arange(485, 515, 0.5)
    ypoints_oiii = 0.97 * np.exp(-0.5 * ((xpoints_oiii - 500.7) / 3.822)**2)
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 6))
    plt.plot(sloan_uprime.waveset.to(u.nm).value, sloan_uprime(sloan_uprime.waveset) * 100, 'b--', label='Baader u\'', alpha=0.5)
    plt.plot(sloan_gprime.waveset.to(u.nm).value, sloan_gprime(sloan_gprime.waveset) * 100, 'g--', label='Baader g\'', alpha=0.5)
    plt.plot(sloan_rprime.waveset.to(u.nm).value, sloan_rprime(sloan_rprime.waveset) * 100, 'r--', label='Baader r\'', alpha=0.5)
    plt.plot(sloan_iprime.waveset.to(u.nm).value, sloan_iprime(sloan_iprime.waveset) * 100, 'm--', label='Baader i\'', alpha=0.5)
    plt.plot(sloan_zprime.waveset.to(u.nm).value, sloan_zprime(sloan_zprime.waveset) * 100, 'c--', label='Baader z\'', alpha=0.5)
    plt.plot(xpoints_oiii, ypoints_oiii * 100, '-.', color='darkorange', label='Baader OIII', alpha=0.5)
    plt.plot(ha_filter.waveset.to(u.nm).value, ha_filter(ha_filter.waveset) * 100, '-.', color='darkred', label=r'Alluxa $H_\alpha$', alpha=0.5)
    plt.plot(atmo_bandpass.waveset.to(u.nm).value, atmo_bandpass(atmo_bandpass.waveset) * 100, 'k:', label='Atmosphere', alpha=0.5)
    plt.plot(qcmos_qe.waveset.to(u.nm).value, qcmos_qe(qcmos_qe.waveset) * 100, 'k', label='Datasheet QE', alpha=0.5)
    qCMOS_meas_QE_wavelengths = np.array([296.7, 400., 500., 550., 600., 640., 700., 800., 900., 1000., 1064.])
    qCMOS_meas_QE = np.array([0.3756, 0.8271, 0.8604, 0.8193, 0.7585, 0.6932, 0.6111, 0.4982, 0.3117, 0.0932, 0.0115])
    xerr = 5
    yerr = 0.05 * 100
    plt.errorbar(qCMOS_meas_QE_wavelengths, qCMOS_meas_QE * 100, xerr=xerr, yerr=yerr, label='Measured QE', fmt='k.', markersize=1)
    # Put legend above the plot
    plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.8, 0.65), fontsize=12)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.xlim(250, 1100)
    plt.ylim(0,100)
    plt.show()
    plt.figure(figsize=(10, 6))
    # Account for loss at fold mirror
    tot_thru_u = baader_uprime * qcmos_qe * atmo_bandpass * 0.05 * 0.9
    tot_thru_g = baader_gprime * qcmos_qe * atmo_bandpass * 0.57 * 0.9
    tot_thru_r = baader_rprime * qcmos_qe * atmo_bandpass * 0.65 * 0.9
    tot_thru_i = baader_iprime * qcmos_qe * atmo_bandpass * 0.28 * 0.9
    tot_thru_z = baader_zprime * qcmos_qe * atmo_bandpass * 0.06 * 0.9
    # tot_thru_u = baader_uprime * qcmos_qe * 0.05
    # tot_thru_g = baader_gprime * qcmos_qe * 0.57
    # tot_thru_r = baader_rprime * qcmos_qe * 0.65
    # tot_thru_i = baader_iprime * qcmos_qe * 0.28
    # tot_thru_z = baader_zprime * qcmos_qe * 0.06
    # tot_thru_u2 = baader_uprime * qcmos_qe * 0.8
    # tot_thru_g2 = baader_gprime * qcmos_qe * 0.8
    # tot_thru_r2 = baader_rprime * qcmos_qe * 0.8
    # tot_thru_i2 = baader_iprime * qcmos_qe * 0.8
    # tot_thru_z2 = baader_zprime * qcmos_qe * 0.8
    oiii_filter = SpectralElement(Empirical1D, 
                                points=xpoints_oiii * 10 * u.AA,
                                lookup_table=ypoints_oiii)
    tot_thru_oiii = oiii_filter * qcmos_qe * atmo_bandpass * 0.57
    tot_thru_halpha = ha_filter * qcmos_qe * atmo_bandpass * 0.65

    plt.plot(tot_thru_u.waveset.to(u.nm).value, tot_thru_u(tot_thru_u.waveset) * 100, 'b--', alpha=0.5)
    plt.plot(tot_thru_g.waveset.to(u.nm).value, tot_thru_g(tot_thru_g.waveset) * 100, 'g--', alpha=0.5)
    plt.plot(tot_thru_r.waveset.to(u.nm).value, tot_thru_r(tot_thru_r.waveset) * 100, 'r--', alpha=0.5)
    plt.plot(tot_thru_i.waveset.to(u.nm).value, tot_thru_i(tot_thru_i.waveset) * 100, 'm--', alpha=0.5)
    plt.plot(tot_thru_z.waveset.to(u.nm).value, tot_thru_z(tot_thru_z.waveset) * 100, 'c--', alpha=0.5)
    # plt.plot(tot_thru_u2.waveset.to(u.nm).value, tot_thru_u2(tot_thru_u2.waveset) * 100, 'b:', alpha=0.5)
    # plt.plot(tot_thru_g2.waveset.to(u.nm).value, tot_thru_g2(tot_thru_g2.waveset) * 100, 'g:', alpha=0.5)
    # plt.plot(tot_thru_r2.waveset.to(u.nm).value, tot_thru_r2(tot_thru_r2.waveset) * 100, 'r:', alpha=0.5)
    # plt.plot(tot_thru_i2.waveset.to(u.nm).value, tot_thru_i2(tot_thru_i2.waveset) * 100, 'm:', alpha=0.5)
    # plt.plot(tot_thru_z2.waveset.to(u.nm).value, tot_thru_z2(tot_thru_z2.waveset) * 100, 'c:', alpha=0.5)
    plt.plot(tot_thru_oiii.waveset.to(u.nm).value, tot_thru_oiii(tot_thru_oiii.waveset) * 100, '-.', color='darkorange', label='Baader OIII', alpha=0.5)
    plt.plot(tot_thru_halpha.waveset.to(u.nm).value, tot_thru_halpha(tot_thru_halpha.waveset) * 100, '-.', color='darkred', label='Halpha', alpha=0.5)
    plt.text(tot_thru_u.pivot().to(u.nm).value, np.max(tot_thru_u(tot_thru_u.waveset)) * 100 + 0.5, "u'", color='b', ha='center')
    plt.text(tot_thru_g.pivot().to(u.nm).value, np.max(tot_thru_g(tot_thru_g.waveset)) * 100 + 0.5, "g'", color='g', ha='center')
    plt.text(tot_thru_r.pivot().to(u.nm).value, np.max(tot_thru_r(tot_thru_r.waveset)) * 100 + 0.5, "r'", color='r', ha='center')
    plt.text(tot_thru_i.pivot().to(u.nm).value, np.max(tot_thru_i(tot_thru_i.waveset)) * 100 + 0.5, "i'", color='m', ha='center')
    plt.text(tot_thru_z.pivot().to(u.nm).value, np.max(tot_thru_z(tot_thru_z.waveset)) * 100 + 0.5, "z'", color='c', ha='center')
    # plt.text(tot_thru_u2.pivot().to(u.nm).value, np.max(tot_thru_u2(tot_thru_u2.waveset)) * 100 + 0.5, "u'", color='b', ha='center')
    # plt.text(tot_thru_g2.pivot().to(u.nm).value, np.max(tot_thru_g2(tot_thru_g2.waveset)) * 100 + 0.5, "g'", color='g', ha='center')
    # plt.text(tot_thru_r2.pivot().to(u.nm).value, np.max(tot_thru_r2(tot_thru_r2.waveset)) * 100 + 0.5, "r'", color='r', ha='center')
    # plt.text(tot_thru_i2.pivot().to(u.nm).value, np.max(tot_thru_i2(tot_thru_i2.waveset)) * 100 + 0.5, "i'", color='m', ha='center')
    # plt.text(tot_thru_z2.pivot().to(u.nm).value, np.max(tot_thru_z2(tot_thru_z2.waveset)) * 100 + 0.5, "z'", color='c', ha='center')
    plt.text(500.7, np.max(tot_thru_oiii(tot_thru_oiii.waveset)) * 100 + 0.5, "OIII", color='darkorange', ha='center')
    plt.text(656.3, np.max(tot_thru_halpha(tot_thru_halpha.waveset)) * 100 + 0.5, r"$H_\alpha$", color='darkred', ha='center')
    plt.xlim(300, 1050)
    plt.ylim(0.1, 45)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    # Make legend showing solid lines are current throughput and dotted lines are potential improved throughput.
    # Use black solid line and black dotted line for legend.
    # plt.plot([], [], 'k:', label='Lightspeed design goal')
    # plt.plot([], [], 'k-', label='proto-Lightspeed')
    # plt.legend()
    plt.show()