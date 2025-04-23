'''GUI for calculating photometry for a ground observatory'''

import os
import tkinter as tk
import pysynphot as S
import numpy as np
import matplotlib.pyplot as plt
from spectra import *
from observatory import Sensor, Telescope, Observatory
from ground_observatory import GroundObservatory
from instruments import sensor_dict_lightspeed, telescope_dict_lightspeed, filter_dict_lightspeed
from tkinter import messagebox

data_folder = os.path.dirname(__file__) + '/../data/'


class MyGUI:
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
        # is the tkinter label itself, the third is the tkinter variable, and the
        # fourth is the tkinter entry box.
        self.sens_options = list(sensor_dict_lightspeed.keys())
        self.sens_vars = {'name': ['Select Sensor'],
                          'pix_size': ['Pixel Size (um)'],
                          'read_noise': ['Read Noise (e-/pix)'],
                          'dark_current': ['Dark Current (e-/pix/s)'],
                          'qe': ['Quantum Efficiency'],
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

        self.plot_qe_button = tk.Button(self.root, text='Plot QE vs. Lambda',
                                    command=self.plot_qe, fg='green')
        self.plot_qe_button.grid(row=7, column=0, columnspan=2, padx=PADX,
                                 pady=PADY)

        # Defining telescope properties
        self.tele_header = tk.Label(self.root, text='Telescope Properties',
                                    font=['Arial', 16, 'bold'])
        self.tele_header.grid(row=8, column=0, columnspan=2, padx=PADX,
                              pady=PADY)
        # Similarly, make a dictionary for telescope properties.
        # ZZZ NEED TO FIGURE OUT WHAT CHANGED TO MAKE RESULTS SLIGHTLY DIFFERENT
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
                              'rho': ['Object-Moon Separation (deg)']}
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
                value.append(tk.DoubleVar())
                value[2].trace_add('write', self.clear_results)
                value.append(tk.Entry(self.root, width=10, textvariable=value[2]))
                value[3].grid(row=i+1, column=3, padx=PADX, pady=PADY)

        # Initializing labels that display results
        self.results_header = tk.Label(self.root, text='General Results',
                                       font=['Arial', 16, 'bold'])
        self.results_header.grid(row=0, column=4, columnspan=1, padx=PADX,
                                 pady=PADY)

        self.run_button_1 = tk.Button(self.root, fg='green',
                                    text='RUN',
                                    command=self.run_calcs)
        self.run_button_1.grid(row=0, column=5, columnspan=1, padx=PADX,
                             pady=PADY)

        self.results_labels = []
        results_label_names = ['Pixel Scale (arcsec/pix)',
                               'Pivot Wavelength (nm)',
                               'PSF FWHM (arcsec)',
                               'Central Pixel Ensquared Energy',
                               'A_eff at Pivot Wavelength (cm^2)', 'Limiting AB magnitude',
                               'Saturating AB magnitude', 'Airmass']
        self.results_data = []
        for i, name in enumerate(results_label_names):
            self.results_labels.append(tk.Label(self.root, text=name))
            self.results_labels[i].grid(row=i+1, column=4, padx=PADX, pady=PADY)
            self.results_data.append(tk.Label(self.root, fg='red'))
            self.results_data[i].grid(row=i+1, column=5, padx=PADX, pady=PADY)

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
                                    'Photometric Precision (ppm)', 'Optimal Aperture Size (pix)',
                                    'Noise Breakdown', 'Turnover Exposure Time (s)']
        self.spec_results_data = []
        for i, name in enumerate(spec_results_label_names):
            self.spec_results_labels.append(tk.Label(self.root, text=name))

            self.spec_results_data.append(tk.Label(self.root, fg='red'))
            # Trying to figure out spacing
            self.spec_results_labels[i].grid(row=i+6, column=6, padx=PADX, pady=PADY)
            self.spec_results_data[i].grid(row=i+6, column=7, padx=PADX, pady=PADY)

        # Make a button to plot mag vs noise
        self.plot_button = tk.Button(self.root, text='Plot Magnitude vs. Photometric Precision',
                                    command=self.plot_mag_vs_noise, fg='green')
        self.plot_button.grid(row=9, column=4, columnspan=2, padx=PADX,
                              pady=PADY)
        # Set default values
        self.sens = sensor_dict_lightspeed['qCMOS']
        self.tele = telescope_dict_lightspeed['Magellan Prototype']
        self.sens_vars['name'][2].set('qCMOS')
        self.tele_vars['name'][2].set('Magellan Prototype')
        self.obs_vars_dict['filter'][2].set('Sloan g\'')
        self.obs_vars_dict['exptime'][2].set(1.0)
        self.obs_vars_dict['num_exposures'][2].set(1)
        self.obs_vars_dict['limiting_snr'][2].set(5.0)
        self.obs_vars_dict['seeing'][2].set(0.5)
        self.obs_vars_dict['zo'][2].set(0.0)
        self.obs_vars_dict['alpha'][2].set(180)
        self.obs_vars_dict['rho'][2].set(45)
        self.root.mainloop()

    def clear_results(self, *args):
        '''Clear results when a new sensor or telescope is selected'''
        for label in self.results_data:
            label.config(text='')
        for label in self.spec_results_data:
            label.config(text='')

    def update_altitude(self, *args):
        '''Update altitude of the telescope based on the selected telescope'''
        altitude_dict = {'Magellan': 2516, 'Palomar': 1712}
        tele_name = self.tele_vars['name'][2].get()
        if 'Magellan' in tele_name:
            self.tele_vars['altitude'][2].set(altitude_dict['Magellan'])
        # Check if name is WINTER or Hale. If either, use palomar alt
        elif 'WINTER' in tele_name or 'Hale' in tele_name:
            self.tele_vars['altitude'][2].set(altitude_dict['Palomar'])

    def update_reimaging_throughput(self, *args):
        '''Update reimaging throughput based on the selected telescope and filter'''
        tele_name = self.tele_vars['name'][2].get()
        filter_name = self.obs_vars_dict['filter'][2].get()
        throughput_dict_prototype = {'Sloan g\'': 0.57, 'Sloan r\'': 0.65,
                                     'Sloan i\'': 0.28, 'Sloan z\'': 0.06,
                                     'Sloan u\'': 0.05}
        throughput_dict_lightspeed = {'Sloan g\'': 0.85, 'Sloan r\'': 0.85,
                                     'Sloan i\'': 0.85, 'Sloan z\'': 0.85,
                                     'Sloan u\'': 0.85}
        if tele_name == 'Magellan Prototype':
            throughput_dict = throughput_dict_prototype
        elif tele_name == 'Magellan Lightspeed':
            throughput_dict = throughput_dict_lightspeed
        else:
            self.obs_vars_dict['reim_throughput'][2].set(1.0)
            return

        if filter_name in throughput_dict.keys():
            throughput = throughput_dict[filter_name]
            self.obs_vars_dict['reim_throughput'][2].set(throughput)
        else:
            self.obs_vars_dict['reim_throughput'][2].set(1.0)

    def set_sens(self, *args):
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
            self.sens_vars['qe'][3].config(textvariable=self.sens_vars['qe'][2])
            self.sens_vars['qe'][3].config(state='normal')
            self.sens_vars['qe'][3].set(np.mean(self.sens.qe.throughput))

    def set_tele(self, *args):
        '''Set the telescope based on the selected telescope name.'''
        self.tele = telescope_dict_lightspeed[self.tele_vars['name'][2].get()]
        self.tele_vars['diam'][2].set(self.tele.diam)
        self.tele_vars['f_num'][2].set(self.tele.f_num)
        self.tele_vars['bandpass'][2].set(self.tele.bandpass)

    def set_obs(self):
        '''Set the observatory when running calculations'''
        if self.sens_vars['qe'][2].get() == 'ARRAY':
            qe = self.sens.qe
        else:
            qe = S.UniformTransmission(float(self.sens_vars['qe'][2].get()))
        sens = Sensor(pix_size=self.sens_vars['pix_size'][2].get(),
                      read_noise=self.sens_vars['read_noise'][2].get(),
                      dark_current=self.sens_vars['dark_current'][2].get(),
                      full_well=self.sens_vars['full_well'][2].get(),
                      qe=qe)
        tele = Telescope(diam=self.tele_vars['diam'][2].get(),
                         f_num=self.tele_vars['f_num'][2].get(),
                         bandpass=self.tele_vars['bandpass'][2].get())
        exposure_time = self.obs_vars_dict['exptime'][2].get()
        num_exposures = int(self.obs_vars_dict['num_exposures'][2].get())
        limiting_snr = self.obs_vars_dict['limiting_snr'][2].get()
        filter_bp = filter_dict_lightspeed[self.obs_vars_dict['filter'][2].get()]
        reimaging_throughput = self.obs_vars_dict['reim_throughput'][2].get()
        filter_bp = S.UniformTransmission(reimaging_throughput) * filter_bp
        seeing_arcsec = self.obs_vars_dict['seeing'][2].get()
        obs_zo = self.obs_vars_dict['zo'][2].get()
        obs_altitude = self.tele_vars['altitude'][2].get()
        obs_alpha = self.obs_vars_dict['alpha'][2].get()
        obs_rho = self.obs_vars_dict['rho'][2].get()
        observatory = GroundObservatory(sens, tele, exposure_time=exposure_time,
                                        num_exposures=num_exposures,
                                        limiting_s_n=limiting_snr,
                                        filter_bandpass=filter_bp,
                                        seeing=seeing_arcsec,
                                        zo=obs_zo, rho=obs_rho,
                                        altitude=obs_altitude,
                                        alpha=obs_alpha)
        return observatory

    def set_spectrum(self):
        '''Set the spectrum based on the selected spectrum type.'''
        if self.flat_spec_bool.get():
            abmag = self.flat_spec_mag.get()
            # Convert to Jansky's; sometimes Pysynphot freaks out when
            # using AB magnitudes.
            fluxdensity_Jy = 10 ** (-0.4 * (abmag - 8.90))
            spectrum = S.FlatSpectrum(fluxdensity=fluxdensity_Jy,
                                      fluxunits='Jy')
            # spectrum.convert('fnu')
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
            limiting_mag = observatory.limiting_mag()
            saturating_mag = observatory.saturating_mag()

            self.results_data[0].config(text=format(observatory.pix_scale, '4.3f'))
            self.results_data[1].config(text=format(observatory.lambda_pivot / 10, '4.1f'))
            psf_fwhm_arcsec = (observatory.psf_fwhm_um() * observatory.pix_scale /
                               observatory.sensor.pix_size)
            self.results_data[2].config(text=format(psf_fwhm_arcsec, '4.3f'))
            self.results_data[3].config(text=format(100 * observatory.central_pix_frac(),
                                                    '4.1f') + '%')
            self.results_data[4].config(text=format(observatory.eff_area_pivot(), '4.2f'))
            self.results_data[5].config(text=format(limiting_mag, '4.3f'))
            self.results_data[6].config(text=format(saturating_mag, '4.3f'))
            self.results_data[7].config(text=format(observatory.airmass, '4.3f'))
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def run_observation(self):
        '''Run calculations for observing a spectrum.'''
        try:
            spectrum = self.set_spectrum()
            observatory = self.set_obs()
            results = observatory.observe(spectrum)
            signal = int(results['signal'])
            noise = int(results['tot_noise'])
            snr = results['signal'] / results['tot_noise']
            phot_prec = 10 ** 6 / snr
            turnover_exp_time = observatory.turnover_exp_time(spectrum)
            self.spec_results_data[0].config(text=format(signal, '4d'))
            self.spec_results_data[1].config(text=format(noise, '4d'))
            self.spec_results_data[2].config(text=format(snr, '4.3f'))
            self.spec_results_data[3].config(text=format(phot_prec, '4.3f'))
            self.spec_results_data[4].config(text=format(results['n_aper'], '2d'))
            noise_str = ('Shot noise: ' + format(results['shot_noise'], '.2f') +
                         '\nDark noise: ' + format(results['dark_noise'], '.2f') +
                         '\nRead noise: ' + format(results['read_noise'], '.2f') +
                         '\nBackground noise: ' + format(results['bkg_noise'], '.2f') +
                         '\nScintillation noise: ' + format(results['scint_noise'], '.2f'))
            self.spec_results_data[5].config(text=noise_str)
            self.spec_results_data[6]. config(text=format(turnover_exp_time, '4.3f'))

        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def plot_qe(self):
        '''Plot the quantum efficiency of the sensor.'''
        qe = self.sens.qe
        # Check if uniform transmission
        if isinstance(qe, S.UniformTransmission):
            wave = np.linspace(200, 1000, 100)
            throughput = np.ones_like(wave) * self.sens_vars['qe'][2].get()
            plt.plot(wave, throughput * 100)
        else:
            plt.plot(qe.wave / 10, qe.throughput * 100)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quantum Efficiency (%)')
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
        img_size = 11
        for i, mag in enumerate(mag_points):
            spectrum = S.FlatSpectrum(mag, fluxunits='abmag')
            results = observatory.observe(spectrum, img_size=img_size)
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


MyGUI()
