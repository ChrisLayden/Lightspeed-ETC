'''GUI for calculating photometry for a ground observatory'''

import os
import copy
import tkinter as tk
from tkinter import messagebox
from synphot import SpectralElement, SourceSpectrum, ConstFlux1D, Empirical1D
import astropy.units as u
from synphot.units import FLAM
import numpy as np
import matplotlib.pyplot as plt
from spectra import *
from observatory import Sensor, Telescope
from ground_observatory import GroundObservatory
from instruments import sensor_dict_lightspeed, telescope_dict_lightspeed, filter_dict_lightspeed, atmo_bandpass

data_folder = os.path.dirname(__file__) + '/../data/'


class MyGUI:
    '''GUI for calculating photometry for a ground observatory'''
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Photometry Calculations')

        PADX = 10
        PADY = 5

        # Defining sensor properties
        self.sens_header = tk.Label(self.root, text='Sensor Properties',
                                    font=['Arial', 16, 'bold'])
        self.sens_header.grid(row=0, column=0, columnspan=2,
                              padx=PADX, pady=PADY)
        # Make a dictionary for sensor properties. Value will be a list,
        # where the first element is the string for the label, the second
        # is the tkinter label, the third is the tkinter variable, and the
        # fourth is the tkinter entry box.
        self.sens_options = list(sensor_dict_lightspeed.keys())
        self.sens_vars = {'name': ['Select Sensor'],
                          'pix_size': ['Pixel Size (um)'],
                          'read_noise': ['Read Noise (e-/pix)'],
                          'dark_current': ['Dark Current (e-/pix/s)'],
                          'qe': ['Quantum Efficiency'],
                          'nonlinearity_scaleup': ['Nonlinearity Scale Factor'],
                          'full_well': ['Full Well Capacity']}
        for i, (key, value) in enumerate(self.sens_vars.items()):
            if key == 'name':
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+1, column=0, padx=PADX, pady=PADY)
                value.append(tk.StringVar())
                value[2].trace_add('write', self.set_sens)
                value[2].trace_add('write', self.clear_results)
                value.append(tk.OptionMenu(self.root, value[2], *self.sens_options))
                value[3].grid(row=i+1, column=1, padx=PADX, pady=PADY)
            else:
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+1, column=0, padx=PADX, pady=PADY)
                value.append(tk.DoubleVar())
                value[2].trace_add('write', self.clear_results)
                value.append(tk.Entry(self.root, width=10, textvariable=value[2]))
                value[3].grid(row=i+1, column=1, padx=PADX, pady=PADY)

        # Defining telescope properties
        self.tele_header = tk.Label(self.root, text='Telescope Properties',
                                    font=['Arial', 16, 'bold'])
        self.tele_header.grid(row=8, column=0, columnspan=2, padx=PADX,
                              pady=PADY)
        # Similarly, make a dictionary for telescope properties.
        self.tele_options = list(telescope_dict_lightspeed.keys())
        self.tele_vars = {'name': ['Select Telescope'],
                               'diam': ['Diameter (cm)'],
                               'f_num': ['F/number'],
                               'bandpass': ['Telescope Throughput'],
                               'altitude': ['Altitude (m)']}
        for i, (key, value) in enumerate(self.tele_vars.items()):
            if key == 'name':
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+9, column=0, padx=PADX, pady=PADY)
                value.append(tk.StringVar())
                value[2].trace_add('write', self.set_tele)
                value[2].trace_add('write', self.clear_results)
                value[2].trace_add('write', self.update_altitude)
                value[2].trace_add('write', self.update_reimaging_throughput)
                value.append(tk.OptionMenu(self.root, value[2], *self.tele_options))
                value[3].grid(row=i+9, column=1, padx=PADX, pady=PADY)
            else:
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+9, column=0, padx=PADX, pady=PADY)
                value.append(tk.DoubleVar())
                value[2].trace_add('write', self.clear_results)
                value.append(tk.Entry(self.root, width=10, textvariable=value[2]))
                value[3].grid(row=i+9, column=1, padx=PADX, pady=PADY)

        # Defining observing properties
        self.obs_header = tk.Label(self.root, text='Observing Properties',
                                   font=['Arial', 16, 'bold'])
        self.obs_header.grid(row=0, column=2, columnspan=2, padx=PADX,
                             pady=PADY)
        # Again, make a dictionary for observing properties.
        self.filter_options = list(filter_dict_lightspeed.keys())
        self.obs_vars_dict = {'exptime': ['Exposure Time (s)'],
                              'num_exposures': ['Exposures in Stack'],
                              'filter': ['Select Filter'],
                              'reim_throughput': ['Reimaging Throughput'],
                              'limiting_snr': ['Limiting SNR'],
                              'seeing': ['Seeing (arcsec)'],
                              'zo': ['Object Zenith Angle (deg)'],
                              'alpha': ['Lunar Phase (deg)'],
                              'rho': ['Object-Moon Separation (deg)'],
                              'aper_rad': ['Aperture Radius (pix)\n(Leave blank for optimal aperture)']}
        for i, (key, value) in enumerate(self.obs_vars_dict.items()):
            if key == 'filter':
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+1, column=2, padx=PADX, pady=PADY)
                value.append(tk.StringVar())
                value[2].trace_add('write', self.clear_results)
                value[2].trace_add('write', self.update_reimaging_throughput)
                value.append(tk.OptionMenu(self.root, value[2], *self.filter_options))
                value[3].grid(row=i+1, column=3, padx=PADX, pady=PADY)
            else:
                value.append(tk.Label(self.root, text=value[0]))
                value[1].grid(row=i+1, column=2, padx=PADX, pady=PADY)
                if key == 'aper_rad':
                    value.append(tk.StringVar())
                else:
                    value.append(tk.DoubleVar())
                value[2].trace_add('write', self.clear_results)
                value.append(tk.Entry(self.root, width=10, textvariable=value[2]))
                value[3].grid(row=i+1, column=3, padx=PADX, pady=PADY)

        self.plot_trans_button = tk.Button(self.root, text='Plot Transmission',
                                    command=self.plot_trans, fg='green')
        self.plot_trans_button.grid(row=i+2, column=2, columnspan=2, padx=PADX,
                                 pady=PADY)

        # Initializing labels that display basic results
        self.results_header = tk.Label(self.root, text='General Results',
                                       font=['Arial', 16, 'bold'])
        self.results_header.grid(row=0, column=4, columnspan=1, padx=PADX,
                                 pady=PADY)

        self.run_button_1 = tk.Button(self.root, fg='green',
                                    text='RUN',
                                    command=self.run_calcs)
        self.run_button_1.grid(row=0, column=5, columnspan=1, padx=PADX,
                             pady=PADY)
        # Make dictionary for results parameters.
        self.basic_results = {'pix_scale': ['Pixel Scale (arcsec/pix)'],
                              'lambda_pivot': ['Pivot Wavelength (Angstrom)'],
                              'psf_fwhm': ['PSF FWHM (arcsec)'],
                              'central_pix_frac': ['Central Pixel Ensquared Energy (%)'],
                              'eff_area_pivot': ['A_eff at Pivot Wavelength (cm^2)'],
                              'limiting_mag': ['Limiting AB magnitude'],
                              'saturating_mag': ['Saturating AB magnitude'],
                              'airmass': ['Airmass'],
                              'zero_point': ['Zero Point AB Magnitude (1 e-/s)'],
                              'exptime_turnover': ['t_exp where bkg noise = read noise (s)']}
        for i, (key, value) in enumerate(self.basic_results.items()):
            value.append(tk.Label(self.root, text=value[0]))
            value[1].grid(row=i+1, column=4, padx=PADX, pady=PADY)
            value.append(tk.DoubleVar())
            value.append(tk.Label(self.root, fg='red'))
            value[3].grid(row=i+1, column=5, padx=PADX, pady=PADY)

        # Set a spectrum to observe
        self.spectrum_header = tk.Label(self.root, text='Spectrum Observation',
                                        font=['Arial', 16, 'bold'])
        self.spectrum_header.grid(row=0, column=6, columnspan=1, padx=PADX,
                                  pady=PADY)
        self.run_button_2 = tk.Button(self.root, fg='green', text='RUN',
                                    command=self.run_observation)
        self.run_button_2.grid(row=0, column=7, columnspan=1, padx=PADX,
                             pady=PADY)
        self.flat_spec_bool = tk.BooleanVar(value=True)
        self.flat_spec_check = tk.Checkbutton(self.root,
                                              text='Flat spectrum at AB mag',
                                              variable=self.flat_spec_bool)
        self.flat_spec_check.grid(row=1, column=6, padx=PADX, pady=PADY)
        self.flat_spec_mag = tk.DoubleVar(value=20.0)
        self.flat_spec_entry = tk.Entry(self.root, width=10,
                                        textvariable=self.flat_spec_mag)
        self.flat_spec_entry.grid(row=1, column=7, padx=PADX, pady=PADY)
        self.bb_spec_bool = tk.BooleanVar()
        self.bb_spec_check = tk.Checkbutton(self.root,
                                            text='Blackbody with Temp (in K)',
                                            variable=self.bb_spec_bool)
        self.bb_spec_check.grid(row=2, column=6, padx=PADX, pady=PADY)
        self.bb_temp = tk.DoubleVar()
        self.bb_spec_entry_1 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_temp)
        self.bb_spec_entry_1.grid(row=2, column=7, padx=PADX, pady=PADY)
        self.bb_dist_label = tk.Label(self.root, text='distance (in Mpc)')
        self.bb_dist_label.grid(row=3, column=6, padx=PADX, pady=PADY)
        self.bb_distance = tk.DoubleVar()
        self.bb_spec_entry_2 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_distance)
        self.bb_spec_entry_2.grid(row=3, column=7, padx=PADX, pady=PADY)
        self.bb_lbol_label = tk.Label(self.root,
                                      text='bolometric luminosity (in erg/s)')
        self.bb_lbol_label.grid(row=4, column=6, padx=PADX, pady=PADY)
        self.bb_lbol = tk.DoubleVar()
        self.bb_spec_entry_3 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_lbol)
        self.bb_spec_entry_3.grid(row=4, column=7, padx=PADX, pady=PADY)
        self.user_spec_bool = tk.BooleanVar()
        self.user_spec_check = tk.Checkbutton(self.root,
                                              text='Spectrum in spectra.py named',
                                              variable=self.user_spec_bool)
        self.user_spec_check.grid(row=5, column=6, padx=PADX, pady=PADY)
        self.user_spec_name = tk.StringVar()
        self.user_spec_entry = tk.Entry(self.root, width=20,
                                        textvariable=self.user_spec_name)
        self.user_spec_entry.grid(row=5, column=7, padx=PADX, pady=PADY)
        self.spec_results_labels = []
        spec_results_label_names = ['Signal (e-)', 'Total Noise (e-)', 'SNR',
                                    'Photometric Precision (ppm)', 'Aperture Size (pix)',
                                    'Noise Breakdown']
        self.spec_results_data = []
        for i, name in enumerate(spec_results_label_names):
            self.spec_results_labels.append(tk.Label(self.root, text=name))
            self.spec_results_data.append(tk.Label(self.root, fg='red'))
            self.spec_results_labels[i].grid(row=i+6, column=6, padx=PADX, pady=PADY)
            self.spec_results_data[i].grid(row=i+6, column=7, padx=PADX, pady=PADY)

        # Make a button to plot mag vs noise
        self.plot_button = tk.Button(self.root, text='Plot Magnitude vs. Photometric Precision',
                                    command=self.plot_mag_vs_noise, fg='green')
        self.plot_button.grid(row=11, column=4, columnspan=2, padx=PADX,
                              pady=PADY)
        self.plot_snr_button = tk.Button(self.root, text='Plot Magnitude vs. SNR',
                                    command=self.plot_mag_vs_snr, fg='green')
        self.plot_snr_button.grid(row=12, column=4, columnspan=2, padx=PADX,
                              pady=PADY)
        # Set default values
        self.sens = sensor_dict_lightspeed['qCMOS']
        self.tele = telescope_dict_lightspeed['Clay (proto-Lightspeed)']
        self.sens_vars['name'][2].set('qCMOS')
        self.tele_vars['name'][2].set('Clay (proto-Lightspeed)')
        self.obs_vars_dict['filter'][2].set('Sloan g\'')
        self.obs_vars_dict['exptime'][2].set(1.0)
        self.obs_vars_dict['num_exposures'][2].set(1)
        self.obs_vars_dict['limiting_snr'][2].set(5.0)
        self.obs_vars_dict['seeing'][2].set(0.5)
        self.obs_vars_dict['zo'][2].set(0.0)
        self.obs_vars_dict['alpha'][2].set(180)
        self.obs_vars_dict['rho'][2].set(45)
        self.root.mainloop()

    def clear_results(self, *_):
        '''Clear results when a new sensor or telescope is selected'''
        labels = [value[3] for value in self.basic_results.values()]
        for label in labels:
            label.config(text='')
        for label in self.spec_results_data:
            label.config(text='')

    def update_altitude(self, *_):
        '''Update altitude of the telescope based on the selected telescope'''
        altitude_dict = {'Clay': 2516, 'Palomar': 1712, 'Keck': 4145}
        tele_name = self.tele_vars['name'][2].get()
        if 'Clay' in tele_name:
            self.tele_vars['altitude'][2].set(altitude_dict['Clay'])
        # Check if name is WINTER or Hale. If either, use palomar alt
        elif 'WINTER' in tele_name or 'Hale' in tele_name:
            self.tele_vars['altitude'][2].set(altitude_dict['Palomar'])
        elif 'Keck' in tele_name:
            self.tele_vars['altitude'][2].set(altitude_dict['Keck'])

    def update_reimaging_throughput(self, *_):
        '''Update reimaging throughput based on the selected telescope and filter'''
        tele_name = self.tele_vars['name'][2].get()
        filter_name = self.obs_vars_dict['filter'][2].get()
        # For white light, make a piecewise throughput
        throughput_dict_prototype = {'Sloan g\'': 0.57, 'Sloan r\'': 0.65,
                                     'Sloan i\'': 0.28, 'Sloan z\'': 0.06,
                                     'Sloan u\'': 0.05, 'Halpha': 0.65,
                                     'Baader OIII': 0.57}
        throughput_dict_lightspeed = {'Sloan g\'': 0.8, 'Sloan r\'': 0.8,
                                     'Sloan i\'': 0.8, 'Sloan z\'': 0.8,
                                     'Sloan u\'': 0.8, 'Halpha': 0.8, 'None': 0.8,
                                     'Baader OIII': 0.8}
        if tele_name == 'Clay (proto-Lightspeed)':
            throughput_dict = throughput_dict_prototype
        elif tele_name == 'Clay (full Lightspeed)':
            throughput_dict = throughput_dict_lightspeed
        else:
            self.obs_vars_dict['reim_throughput'][2].set(1.0)
            return

        if filter_name in throughput_dict.keys():
            throughput = throughput_dict[filter_name]
            self.obs_vars_dict['reim_throughput'][2].set(throughput)
        else:
            self.obs_vars_dict['reim_throughput'][2].set(1.0)

    def set_sens(self, *_):
        '''Set the sensor based on the selected sensor name.'''
        self.sens = sensor_dict_lightspeed[self.sens_vars['name'][2].get()]
        self.sens_vars['pix_size'][2].set(self.sens.pix_size)
        self.sens_vars['read_noise'][2].set(self.sens.read_noise)
        self.sens_vars['dark_current'][2].set(self.sens.dark_current)
        self.sens_vars['full_well'][2].set(self.sens.full_well)
        if self.sens_vars['name'][2].get() != 'Define New Sensor':
            self.sens_vars['qe'][2] = tk.StringVar()
            self.sens_vars['qe'][2].set('ARRAY')
            self.sens_vars['qe'][3].config(textvariable=self.sens_vars['qe'][2])
            self.sens_vars['qe'][3].config(state='disabled')
        else:
            self.sens_vars['qe'][2] = tk.DoubleVar()
            self.sens_vars['qe'][2].set(self.sens.qe(5000 * u.AA).to_value())
            self.sens_vars['qe'][3].config(textvariable=self.sens_vars['qe'][2])
            self.sens_vars['qe'][3].config(state='normal')
        if self.sens.nonlinearity_scaleup is not None:
            self.sens_vars['nonlinearity_scaleup'][2] = tk.StringVar()
            self.sens_vars['nonlinearity_scaleup'][2].set('ARRAY')
            self.sens_vars['nonlinearity_scaleup'][3].config(textvariable=self.sens_vars['nonlinearity_scaleup'][2])
            self.sens_vars['nonlinearity_scaleup'][3].config(state='disabled')
        else:
            self.sens_vars['nonlinearity_scaleup'][2] = tk.DoubleVar()
            self.sens_vars['nonlinearity_scaleup'][2].set(1.0)
            self.sens_vars['nonlinearity_scaleup'][3].config(textvariable=self.sens_vars['nonlinearity_scaleup'][2])
            self.sens_vars['nonlinearity_scaleup'][3].config(state='normal')

    def set_tele(self, *_):
        '''Set the telescope based on the selected telescope name.'''
        self.tele = telescope_dict_lightspeed[self.tele_vars['name'][2].get()]
        self.tele_vars['diam'][2].set(self.tele.diam)
        self.tele_vars['f_num'][2].set(self.tele.f_num)
        # For constant transmission, get the amplitude value
        if hasattr(self.tele.bandpass, 'model') and hasattr(self.tele.bandpass.model, 'amplitude'):
            self.tele_vars['bandpass'][2].set(np.round(self.tele.bandpass.model.amplitude.value,3))
        else:
            self.tele_vars['bandpass'][2].set(1.0)

    def set_obs(self):
        '''Set the observatory when running calculations'''
        if self.sens_vars['qe'][2].get() == 'ARRAY':
            qe = self.sens.qe
        else:
            qe = SpectralElement(ConstFlux1D, amplitude=float(self.sens_vars['qe'][2].get()))
        if self.sens_vars['nonlinearity_scaleup'][2].get() == 'ARRAY':
            nonlinearity_scaleup = self.sens.nonlinearity_scaleup
        else:
            nonlinearity_scaleup = None
        sens = Sensor(pix_size=self.sens_vars['pix_size'][2].get(),
                      read_noise=self.sens_vars['read_noise'][2].get(),
                      dark_current=self.sens_vars['dark_current'][2].get(),
                      full_well=self.sens_vars['full_well'][2].get(),
                      qe=qe, nonlinearity_scaleup=nonlinearity_scaleup)
        tele = Telescope(diam=self.tele_vars['diam'][2].get(),
                         f_num=self.tele_vars['f_num'][2].get(),
                         bandpass=self.tele_vars['bandpass'][2].get())
        exposure_time = self.obs_vars_dict['exptime'][2].get()
        num_exposures = int(self.obs_vars_dict['num_exposures'][2].get())
        limiting_snr = self.obs_vars_dict['limiting_snr'][2].get()
        filter_bp = filter_dict_lightspeed[self.obs_vars_dict['filter'][2].get()]
        reimaging_throughput = self.obs_vars_dict['reim_throughput'][2].get()
        reimaging_bp = SpectralElement(ConstFlux1D, amplitude=reimaging_throughput)
        total_filter_bp = filter_bp * reimaging_bp
        seeing_arcsec = self.obs_vars_dict['seeing'][2].get()
        obs_zo = self.obs_vars_dict['zo'][2].get()
        obs_altitude = self.tele_vars['altitude'][2].get()
        obs_alpha = self.obs_vars_dict['alpha'][2].get()
        obs_rho = self.obs_vars_dict['rho'][2].get()
        obs_aper_rad = self.obs_vars_dict['aper_rad'][2].get()
        obs_aper_rad = None if str(obs_aper_rad) in ['', 'None'] else obs_aper_rad
        observatory = GroundObservatory(sens, tele, exposure_time=exposure_time,
                                        num_exposures=num_exposures,
                                        limiting_s_n=limiting_snr,
                                        filter_bandpass=total_filter_bp,
                                        seeing=seeing_arcsec,
                                        zo=obs_zo, rho=obs_rho,
                                        altitude=obs_altitude,
                                        alpha=obs_alpha,
                                        aper_radius=obs_aper_rad)
        return observatory

    def set_spectrum(self):
        '''Set the spectrum based on the selected spectrum type.'''
        if self.flat_spec_bool.get():
            abmag = self.flat_spec_mag.get()
            # Convert to Jansky's; sometimes Pysynphot freaks out when
            # using AB magnitudes.
            fluxdensity_jy = 10 ** (-0.4 * (abmag - 8.90))
            spectrum = SourceSpectrum(ConstFlux1D, amplitude=fluxdensity_jy * u.Jy)
        elif self.bb_spec_bool.get():
            temp = self.bb_temp.get()
            distance = self.bb_distance.get()
            l_bol = self.bb_lbol.get()
            spectrum = blackbody_spec(temp, distance, l_bol)
        elif self.user_spec_bool.get():
            spectrum_name = self.user_spec_name.get()
            spectrum = eval(spectrum_name)
        else:
            raise ValueError('No spectrum specified')
        return spectrum

    def run_calcs(self):
        '''Run calculations for basic observing properties.'''
        try:
            observatory = self.set_obs()
            self.basic_results['pix_scale'][2] = observatory.pix_scale
            self.basic_results['lambda_pivot'][2] = observatory.lambda_pivot.to(u.AA).value
            fwhm_arcsec = (observatory.psf_fwhm_um() * observatory.pix_scale /
                           observatory.sensor.pix_size)
            self.basic_results['psf_fwhm'][2] = fwhm_arcsec
            self.basic_results['central_pix_frac'][2] = observatory.central_pix_frac() * 100
            self.basic_results['eff_area_pivot'][2] = observatory.eff_area(observatory.lambda_pivot).value
            self.basic_results['limiting_mag'][2] = observatory.limiting_mag()
            self.basic_results['saturating_mag'][2] = observatory.saturating_mag()
            self.basic_results['airmass'][2] = observatory.airmass
            self.basic_results['zero_point'][2] = observatory.zero_point_mag()
            self.basic_results['exptime_turnover'][2] = observatory.turnover_exp_time()
            for key, value in self.basic_results.items():
                if key in ['lambda_pivot', 'eff_area_pivot']:
                    value[3].config(text=format(value[2], '5.0f'))
                else:
                    value[3].config(text=format(value[2], '4.3f'))
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def run_observation(self):
        '''Run calculations for observing a spectrum.'''
        try:
            spectrum = self.set_spectrum()
            observatory = self.set_obs()
            results = observatory.observe(spectrum)
            signal = results['signal']
            noise = results['tot_noise']
            snr = results['signal'] / results['tot_noise']
            phot_prec = 10 ** 6 / snr
            self.spec_results_data[0].config(text=format(signal, '3.2f'))
            self.spec_results_data[1].config(text=format(noise, '3.2f'))
            self.spec_results_data[2].config(text=format(snr, '4.3f'))
            self.spec_results_data[3].config(text=format(phot_prec, '4.3f'))
            self.spec_results_data[4].config(text=format(results['n_aper'], '2d'))
            noise_str = ('Shot noise: ' + format(results['shot_noise'], '.2f') +
                         '\nDark noise: ' + format(results['dark_noise'], '.2f') +
                         '\nRead noise: ' + format(results['read_noise'], '.2f') +
                         '\nBackground noise: ' + format(results['bkg_noise'], '.2f') +
                         '\nScintillation noise: ' + format(results['scint_noise'], '.2f'))
            self.spec_results_data[5].config(text=noise_str)
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def plot_trans(self):
        '''Plot the transmission of the instrument.'''
        obs = self.set_obs()
        sens_bp = obs.sensor.qe
        tele_bp = obs.telescope.bandpass
        filter_bp = filter_dict_lightspeed[self.obs_vars_dict['filter'][2].get()]
        reimaging_throughput = self.obs_vars_dict['reim_throughput'][2].get()
        reimaging_bp = SpectralElement(ConstFlux1D, amplitude=reimaging_throughput)
        atmo_throughput_with_airmass = atmo_bandpass(atmo_bandpass.waveset) ** obs.airmass
        atmo_bp = SpectralElement(Empirical1D, points=atmo_bandpass.waveset,
                                lookup_table=atmo_throughput_with_airmass)
        total_bp = obs.bandpass
        bps_to_plot = [sens_bp, filter_bp, reimaging_bp, tele_bp, atmo_bp, total_bp]
        bp_names = ['Sensor QE', 'Filter', 'Reimaging Optics', 'Telescope', 'Atmosphere', 'Total']
        linestyles = ['--', ':', '--', ':', '-.', '-']
        for bp, name, ls in zip(bps_to_plot, bp_names, linestyles):
            if hasattr(bp, 'model') and hasattr(bp.model, 'amplitude') and bp.waveset is None:
                # Uniform transmission
                wave = np.linspace(250, 1100, 100)
                throughput = np.ones_like(wave) * bp.model.amplitude.value
                plt.plot(wave, throughput * 100, ls, label=name)
            else:
                # Array-based bandpass
                wave_nm = bp.waveset.to(u.nm).value
                throughput = bp(bp.waveset)
                plt.plot(wave_nm, throughput * 100, ls, label=name)
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission (%)')
        plt.show()

    def plot_mag_vs_noise(self):
        '''Plot the photometric precision vs. AB magnitude.'''
        mag_points = np.linspace(10, 28, 15)
        ppm_points = np.zeros_like(mag_points)
        ppm_points_source = np.zeros_like(mag_points)
        ppm_points_read = np.zeros_like(mag_points)
        ppm_points_bkg = np.zeros_like(mag_points)
        ppm_points_dc = np.zeros_like(mag_points)
        ppm_points_scint = np.zeros_like(mag_points)
        observatory = self.set_obs()
        for i, mag in enumerate(mag_points):
            spectrum = SourceSpectrum(ConstFlux1D, amplitude=mag * u.ABmag)
            results = observatory.observe(spectrum)
            snr = results['signal'] / results['tot_noise']
            phot_prec = 10 ** 6 / snr
            ppm_points[i] = phot_prec
            ppm_points_source[i] = 10 ** 6 * results['shot_noise'] / results['signal']
            ppm_points_read[i] = 10 ** 6 * results['read_noise'] / results['signal']
            ppm_points_bkg[i] = 10 ** 6 * results['bkg_noise'] / results['signal']
            ppm_points_dc[i] = 10 ** 6 * results['dark_noise'] / results['signal']
            ppm_points_scint[i] = 10 ** 6 * results['scint_noise'] / results['signal']
            img_size = results['img_size']
        plt.plot(mag_points, ppm_points, label='Total Noise')
        plt.plot(mag_points, ppm_points_source, label='Shot Noise')
        plt.plot(mag_points, ppm_points_read, label='Read Noise')
        plt.plot(mag_points, ppm_points_bkg, label='Background Noise')
        plt.plot(mag_points, ppm_points_dc, label='Dark Current Noise')
        plt.plot(mag_points, ppm_points_scint, label='Scintillation Noise')
        ppm_threshold = 1e6 / self.obs_vars_dict['limiting_snr'][2].get()
        plt.fill_between(np.linspace(10, 30, 10), ppm_threshold, 2e6, color='red', alpha=0.1,
                         label='Non-detection')
        plt.xlim(10, 28)
        # Just set top upper limit
        plt.ylim(1, 2e6)
        plt.xlabel('AB Magnitude')
        plt.ylabel('Photometric Precision (ppm)')
        plt.yscale('log')
        plt.legend()
        # Make title text with telescope name, bandbpass, and exposure time
        tele_name = self.tele_vars['name'][2].get()
        bandpass = self.obs_vars_dict['filter'][2].get()
        exposure_time = self.obs_vars_dict['exptime'][2].get()
        title = f'{tele_name}, {bandpass}, t_exp={exposure_time}s'
        plt.title(title)
        plt.show()

    def plot_mag_vs_snr(self):
        '''Plot the photometric precision vs. AB magnitude.'''
        # Increase font size
        plt.rcParams.update({'font.size': 14})
        # Make figure wider
        plt.figure(figsize=(10, 6))
        # Leave space for legend on the right side
        plt.subplots_adjust(right=0.75)
        exp_time_vals = [0.0002, 0.001, 0.004]
        
        for j, exp_time in enumerate(exp_time_vals):
            self.obs_vars_dict['exptime'][2].set(exp_time)
            num_exp = 3 * 3600 / exp_time
            self.obs_vars_dict['num_exposures'][2].set(num_exp)
            observatory = self.set_obs()
            mag_points = np.linspace(15, 31, 17)
            snr_points = np.zeros_like(mag_points)
            for i, mag in enumerate(mag_points):
                spectrum = SourceSpectrum(ConstFlux1D, amplitude=mag * u.ABmag)
                results = observatory.observe(spectrum)
                snr = results['signal'] / results['tot_noise']
                snr_points[i] = snr
            if j == 0:
                plt.plot(mag_points, snr_points, 'k-', label=r'5000 Hz')
            elif j == 1:
                plt.plot(mag_points, snr_points, 'k--', label=r'1000 Hz')
            else:
                plt.plot(mag_points, snr_points, 'k-.', label=r'250 Hz')
        # snr_threshold = self.obs_vars_dict['limiting_snr'][2].get()
        # plt.fill_between(np.linspace(10, 30, 10), 0, snr_threshold, color='red', alpha=0.1,
        #                  label='Non-detection')
        # Have colors advance from blue to red as magnitude increases
        plt.axvline(x=16.5, color='magenta', ls='-.', label='SAXJ1808-3658')
        plt.axvline(x=16.6, color='purple', ls=':', label='Crab')
        plt.axvline(x=17.5, color='blue', ls='-.', label='J1023+0038')
        # plt.axvline(x=21, color='cyan', ls='-.', label='J0437−4715')
        plt.axvline(x=22.1, color='lime', ls=':', label='B1509-58')
        plt.axvline(x=22.5, color='darkgreen', ls=':', label='B0540-69')
        plt.axvline(x=23.6, color='orange', ls=':', label='Vela')
        plt.axvline(x=25.0, color='red', ls=':', label='B0656+14')
        plt.axvline(x=25.5, color='brown', ls=':', label='Geminga')
        plt.axvline(x=26.0, color='cyan', ls='-.', label='J2339-0533')
        plt.axhline(y=10, color='gray', ls='-', alpha=0.5)
        plt.axhline(y=100, color='gray', ls='-', alpha=0.5)
        plt.axhline(y=1000, color='gray', ls='-', alpha=0.5)
        plt.axhline(y=10000, color='gray', ls='-', alpha=0.5)
        plt.xlim(15, 30)
        # Set lower ylim
        plt.ylim(1, 1e5)
        plt.xlabel('AB Magnitude')
        plt.ylabel('White light SNR in 3 hours')
        plt.yscale('log')
        # Put legend to right side of plot OUTSIDE OF PLOT
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        plt.show()


MyGUI()