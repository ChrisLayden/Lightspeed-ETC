'''GUI for calculating photometry for a ground observatory'''

import os
import tkinter as tk
from tkinter import messagebox
from synphot import SpectralElement, SourceSpectrum, ConstFlux1D, Empirical1D
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from spectra import blackbody_spec
from observatory import Sensor, Telescope
from ground_observatory import GroundObservatory
from instruments import (sensor_dict_lightspeed, telescope_dict_lightspeed,
                        filter_dict_lightspeed, atmo_bandpass, lightspeed_thru,
                        throughput_proto)

data_folder = os.path.dirname(__file__) + '/../data/'


class MyGUI:
    '''GUI for calculating photometry for a ground observatory'''
    
    # Layout constants
    PADX = 10
    PADY = 5
    FONT_HEADER = ('Arial', 16, 'bold')
    COLOR_BUTTON = 'green'
    COLOR_RESULT = 'red'
    
    # Observatory constants
    ALTITUDES = {'Clay': 2516, 'Palomar': 1712, 'Keck': 4145}
    
    # Instrument throughput SpectralElements (None means no additional throughput)
    INSTRUMENT_THROUGHPUT = {
        'Clay (proto-Lightspeed)': throughput_proto,
        'Clay (full Lightspeed)': lightspeed_thru,
        'Keck I': lightspeed_thru,
        'Keck II': lightspeed_thru,
        'Clay Prime Focus': None,
        'Hale': None,
        'WINTER': None,
    }
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Photometry Calculations')
        
        # Store current instrument throughput
        self.current_instrument_thru = None
        
        self._create_sensor_section()
        self._create_telescope_section()
        self._create_observing_section()
        self._create_results_section()
        self._create_spectrum_section()
        self._create_plot_buttons()
        self._set_defaults()
        
        self.root.mainloop()
    
    def _create_sensor_section(self):
        '''Create sensor properties section'''
        row_start, col_start = 0, 0
        
        tk.Label(self.root, text='Sensor Properties', font=self.FONT_HEADER).grid(
            row=row_start, column=col_start, columnspan=2,
            padx=self.PADX, pady=self.PADY)
        
        self.sens_options = list(sensor_dict_lightspeed.keys())
        self.sens_vars = {
            'name': ['Select Sensor'],
            'pix_size': ['Pixel Size (um)'],
            'read_noise': ['Read Noise (e-/pix)'],
            'dark_current': ['Dark Current (e-/pix/s)'],
            'qe': ['Quantum Efficiency'],
            'nonlinearity_scaleup': ['Nonlinearity Scale Factor'],
            'full_well': ['Full Well Capacity']
        }
        
        for i, (key, value) in enumerate(self.sens_vars.items()):
            row = row_start + i + 1
            label = tk.Label(self.root, text=value[0])
            label.grid(row=row, column=col_start, padx=self.PADX, pady=self.PADY)
            value.append(label)
            
            if key == 'name':
                var = tk.StringVar()
                var.trace_add('write', self.set_sens)
                var.trace_add('write', self.clear_results)
                widget = tk.OptionMenu(self.root, var, *self.sens_options)
            else:
                var = tk.DoubleVar()
                var.trace_add('write', self.clear_results)
                widget = tk.Entry(self.root, width=10, textvariable=var)
            
            value.append(var)
            value.append(widget)
            widget.grid(row=row, column=col_start + 1, padx=self.PADX, pady=self.PADY)
    
    def _create_telescope_section(self):
        '''Create telescope properties section'''
        row_start, col_start = 8, 0
        
        tk.Label(self.root, text='Telescope Properties', font=self.FONT_HEADER).grid(
            row=row_start, column=col_start, columnspan=2,
            padx=self.PADX, pady=self.PADY)
        
        self.tele_options = list(telescope_dict_lightspeed.keys())
        self.tele_vars = {
            'name': ['Select Telescope'],
            'diam': ['Diameter (cm)'],
            'f_num': ['F/number'],
            'bandpass': ['Telescope Throughput'],
            'altitude': ['Altitude (m)']
        }
        
        for i, (key, value) in enumerate(self.tele_vars.items()):
            row = row_start + i + 1
            label = tk.Label(self.root, text=value[0])
            label.grid(row=row, column=col_start, padx=self.PADX, pady=self.PADY)
            value.append(label)
            
            if key == 'name':
                var = tk.StringVar()
                var.trace_add('write', self.set_tele)
                var.trace_add('write', self.clear_results)
                var.trace_add('write', self.update_altitude)
                var.trace_add('write', self.update_instrument_throughput)
                widget = tk.OptionMenu(self.root, var, *self.tele_options)
            else:
                var = tk.DoubleVar()
                var.trace_add('write', self.clear_results)
                widget = tk.Entry(self.root, width=10, textvariable=var)
            
            value.append(var)
            value.append(widget)
            widget.grid(row=row, column=col_start + 1, padx=self.PADX, pady=self.PADY)
    
    def _create_observing_section(self):
        '''Create observing properties section'''
        row_start, col_start = 0, 2
        
        tk.Label(self.root, text='Observing Properties', font=self.FONT_HEADER).grid(
            row=row_start, column=col_start, columnspan=2,
            padx=self.PADX, pady=self.PADY)
        
        self.filter_options = list(filter_dict_lightspeed.keys())
        self.obs_vars_dict = {
            'exptime': ['Exposure Time (s)'],
            'num_exposures': ['Exposures in Stack'],
            'filter': ['Select Filter'],
            'instrument_throughput': ['Instrument Throughput'],
            'limiting_snr': ['Limiting SNR'],
            'seeing': ['Seeing (arcsec)'],
            'zo': ['Object Zenith Angle (deg)'],
            'alpha': ['Lunar Phase (deg)'],
            'rho': ['Object-Moon Separation (deg)'],
            'aper_rad': ['Aperture Radius (pix)\n(Leave blank for optimal aperture)']
        }
        
        for i, (key, value) in enumerate(self.obs_vars_dict.items()):
            row = row_start + i + 1
            label = tk.Label(self.root, text=value[0])
            label.grid(row=row, column=col_start, padx=self.PADX, pady=self.PADY)
            value.append(label)
            
            if key == 'filter':
                var = tk.StringVar()
                var.trace_add('write', self.clear_results)
                widget = tk.OptionMenu(self.root, var, *self.filter_options)
            elif key in ['aper_rad', 'instrument_throughput']:
                var = tk.StringVar()
                var.trace_add('write', self.clear_results)
                widget = tk.Entry(self.root, width=10, textvariable=var)
            else:
                var = tk.DoubleVar()
                var.trace_add('write', self.clear_results)
                widget = tk.Entry(self.root, width=10, textvariable=var)
            
            value.append(var)
            value.append(widget)
            widget.grid(row=row, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        tk.Button(self.root, text='Plot Transmission', command=self.plot_trans,
                 fg=self.COLOR_BUTTON).grid(
            row=len(self.obs_vars_dict) + 1, column=col_start, columnspan=2,
            padx=self.PADX, pady=self.PADY)
    
    def _create_results_section(self):
        '''Create general results section'''
        row_start, col_start = 0, 4
        
        tk.Label(self.root, text='General Results', font=self.FONT_HEADER).grid(
            row=row_start, column=col_start, padx=self.PADX, pady=self.PADY)
        
        tk.Button(self.root, text='RUN', command=self.run_calcs,
                 fg=self.COLOR_BUTTON).grid(
            row=row_start, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        self.basic_results = {
            'pix_scale': ['Pixel Scale (arcsec/pix)'],
            'lambda_pivot': ['Pivot Wavelength (Angstrom)'],
            'psf_fwhm': ['PSF FWHM (arcsec)'],
            'central_pix_frac': ['Central Pixel Ensquared Energy (%)'],
            'eff_area_pivot': ['A_eff at Pivot Wavelength (cm^2)'],
            'limiting_mag': ['Limiting AB magnitude'],
            'saturating_mag': ['Saturating AB magnitude'],
            'airmass': ['Airmass'],
            'zero_point': ['Zero Point AB Magnitude (1 e-/s)'],
            'exptime_turnover': ['t_exp where bkg noise = read noise (s)']
        }
        
        for i, (key, value) in enumerate(self.basic_results.items()):
            row = row_start + i + 1
            label = tk.Label(self.root, text=value[0])
            label.grid(row=row, column=col_start, padx=self.PADX, pady=self.PADY)
            value.append(label)
            value.append(tk.DoubleVar())
            result_label = tk.Label(self.root, fg=self.COLOR_RESULT)
            result_label.grid(row=row, column=col_start + 1, padx=self.PADX, pady=self.PADY)
            value.append(result_label)
    
    def _create_spectrum_section(self):
        '''Create spectrum observation section'''
        row_start, col_start = 0, 6
        
        tk.Label(self.root, text='Spectrum Observation', font=self.FONT_HEADER).grid(
            row=row_start, column=col_start, padx=self.PADX, pady=self.PADY)
        
        tk.Button(self.root, text='RUN', command=self.run_observation,
                 fg=self.COLOR_BUTTON).grid(
            row=row_start, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        # Flat spectrum option
        self.flat_spec_bool = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text='Flat spectrum at AB mag',
                      variable=self.flat_spec_bool).grid(
            row=1, column=col_start, padx=self.PADX, pady=self.PADY)
        self.flat_spec_mag = tk.DoubleVar(value=20.0)
        tk.Entry(self.root, width=10, textvariable=self.flat_spec_mag).grid(
            row=1, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        # Blackbody spectrum option
        self.bb_spec_bool = tk.BooleanVar()
        tk.Checkbutton(self.root, text='Blackbody with Temp (in K)',
                      variable=self.bb_spec_bool).grid(
            row=2, column=col_start, padx=self.PADX, pady=self.PADY)
        self.bb_temp = tk.DoubleVar()
        tk.Entry(self.root, width=10, textvariable=self.bb_temp).grid(
            row=2, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        tk.Label(self.root, text='distance (in Mpc)').grid(
            row=3, column=col_start, padx=self.PADX, pady=self.PADY)
        self.bb_distance = tk.DoubleVar()
        tk.Entry(self.root, width=10, textvariable=self.bb_distance).grid(
            row=3, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        tk.Label(self.root, text='bolometric luminosity (in erg/s)').grid(
            row=4, column=col_start, padx=self.PADX, pady=self.PADY)
        self.bb_lbol = tk.DoubleVar()
        tk.Entry(self.root, width=10, textvariable=self.bb_lbol).grid(
            row=4, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        # User spectrum option
        self.user_spec_bool = tk.BooleanVar()
        tk.Checkbutton(self.root, text='Spectrum in spectra.py named',
                      variable=self.user_spec_bool).grid(
            row=5, column=col_start, padx=self.PADX, pady=self.PADY)
        self.user_spec_name = tk.StringVar()
        tk.Entry(self.root, width=20, textvariable=self.user_spec_name).grid(
            row=5, column=col_start + 1, padx=self.PADX, pady=self.PADY)
        
        # Results labels
        spec_results_label_names = [
            'Signal (e-)', 'Total Noise (e-)', 'SNR',
            'Photometric Precision (ppm)', 'Aperture Size (pix)',
            'Noise Breakdown'
        ]
        self.spec_results_labels = []
        self.spec_results_data = []
        
        for i, name in enumerate(spec_results_label_names):
            label = tk.Label(self.root, text=name)
            label.grid(row=i + 6, column=col_start, padx=self.PADX, pady=self.PADY)
            self.spec_results_labels.append(label)
            
            data_label = tk.Label(self.root, fg=self.COLOR_RESULT)
            data_label.grid(row=i + 6, column=col_start + 1, padx=self.PADX, pady=self.PADY)
            self.spec_results_data.append(data_label)
    
    def _create_plot_buttons(self):
        '''Create plotting buttons'''
        tk.Button(self.root, text='Plot Magnitude vs. Photometric Precision',
                 command=self.plot_mag_vs_noise, fg=self.COLOR_BUTTON).grid(
            row=11, column=4, columnspan=2, padx=self.PADX, pady=self.PADY)
        
        tk.Button(self.root, text='Plot Magnitude vs. SNR',
                 command=self.plot_mag_vs_snr, fg=self.COLOR_BUTTON).grid(
            row=12, column=4, columnspan=2, padx=self.PADX, pady=self.PADY)
    
    def _set_defaults(self):
        '''Set default values for all parameters'''
        self.sens = sensor_dict_lightspeed['qCMOS']
        self.tele = telescope_dict_lightspeed['Clay (proto-Lightspeed)']
        
        self.sens_vars['name'][2].set('qCMOS')
        self.tele_vars['name'][2].set('Clay (proto-Lightspeed)')
        self.obs_vars_dict['filter'][2].set("Baader g'")
        self.obs_vars_dict['exptime'][2].set(1.0)
        self.obs_vars_dict['num_exposures'][2].set(1)
        self.obs_vars_dict['limiting_snr'][2].set(5.0)
        self.obs_vars_dict['seeing'][2].set(0.5)
        self.obs_vars_dict['zo'][2].set(0.0)
        self.obs_vars_dict['alpha'][2].set(180)
        self.obs_vars_dict['rho'][2].set(45)
    
    def clear_results(self, *_):
        '''Clear results when a new sensor or telescope is selected'''
        for value in self.basic_results.values():
            value[3].config(text='')
        for label in self.spec_results_data:
            label.config(text='')
    
    def update_altitude(self, *_):
        '''Update altitude of the telescope based on the selected telescope'''
        tele_name = self.tele_vars['name'][2].get()
        
        if 'Clay' in tele_name:
            altitude = self.ALTITUDES['Clay']
        elif 'WINTER' in tele_name or 'Hale' in tele_name:
            altitude = self.ALTITUDES['Palomar']
        elif 'Keck' in tele_name:
            altitude = self.ALTITUDES['Keck']
        else:
            return
        
        self.tele_vars['altitude'][2].set(altitude)
    
    def update_instrument_throughput(self, *_):
        '''Update instrument throughput based on the selected telescope'''
        tele_name = self.tele_vars['name'][2].get()
        
        # Find matching instrument throughput
        self.current_instrument_thru = None
        for key, thru in self.INSTRUMENT_THROUGHPUT.items():
            if key in tele_name:
                self.current_instrument_thru = thru
                break
        
        # Update the display
        if self.current_instrument_thru is not None:
            self.obs_vars_dict['instrument_throughput'][2].set('ARRAY')
            self.obs_vars_dict['instrument_throughput'][3].config(state='disabled')
        else:
            self.obs_vars_dict['instrument_throughput'][2].set('1.0')
            self.obs_vars_dict['instrument_throughput'][3].config(state='normal')
    
    def set_sens(self, *_):
        '''Set the sensor based on the selected sensor name'''
        sens_name = self.sens_vars['name'][2].get()
        self.sens = sensor_dict_lightspeed[sens_name]
        
        self.sens_vars['pix_size'][2].set(self.sens.pix_size)
        self.sens_vars['read_noise'][2].set(self.sens.read_noise)
        self.sens_vars['dark_current'][2].set(self.sens.dark_current)
        self.sens_vars['full_well'][2].set(self.sens.full_well)
        
        # Handle QE display
        if sens_name != 'Define New Sensor':
            self.sens_vars['qe'][2] = tk.StringVar()
            self.sens_vars['qe'][2].set('ARRAY')
            self.sens_vars['qe'][3].config(textvariable=self.sens_vars['qe'][2], state='disabled')
        else:
            self.sens_vars['qe'][2] = tk.DoubleVar()
            self.sens_vars['qe'][2].set(self.sens.qe(5000 * u.AA).to_value())
            self.sens_vars['qe'][3].config(textvariable=self.sens_vars['qe'][2], state='normal')
        
        # Handle nonlinearity scaleup display
        if self.sens.nonlinearity_scaleup is not None:
            self.sens_vars['nonlinearity_scaleup'][2] = tk.StringVar()
            self.sens_vars['nonlinearity_scaleup'][2].set('ARRAY')
            self.sens_vars['nonlinearity_scaleup'][3].config(
                textvariable=self.sens_vars['nonlinearity_scaleup'][2], state='disabled')
        else:
            self.sens_vars['nonlinearity_scaleup'][2] = tk.DoubleVar()
            self.sens_vars['nonlinearity_scaleup'][2].set(1.0)
            self.sens_vars['nonlinearity_scaleup'][3].config(
                textvariable=self.sens_vars['nonlinearity_scaleup'][2], state='normal')
    
    def set_tele(self, *_):
        '''Set the telescope based on the selected telescope name'''
        self.tele = telescope_dict_lightspeed[self.tele_vars['name'][2].get()]
        
        self.tele_vars['diam'][2].set(self.tele.diam)
        self.tele_vars['f_num'][2].set(self.tele.f_num)
        
        # Get bandpass value
        if hasattr(self.tele.bandpass, 'model') and hasattr(self.tele.bandpass.model, 'amplitude'):
            bandpass_val = np.round(self.tele.bandpass.model.amplitude.value, 3)
        else:
            bandpass_val = 1.0
        self.tele_vars['bandpass'][2].set(bandpass_val)
    
    def get_instrument_throughput_bp(self):
        '''Get the instrument throughput as a SpectralElement, or None if unity'''
        thru_val = self.obs_vars_dict['instrument_throughput'][2].get()
        
        if thru_val == 'ARRAY' and self.current_instrument_thru is not None:
            return self.current_instrument_thru
        else:
            try:
                scalar_val = float(thru_val)
            except ValueError:
                scalar_val = 1.0
            # Return None for unity throughput to skip multiplication
            if scalar_val == 1.0:
                return None
            return SpectralElement(ConstFlux1D, amplitude=scalar_val)
    
    def set_obs(self):
        '''Set the observatory when running calculations'''
        # Get sensor QE
        if self.sens_vars['qe'][2].get() == 'ARRAY':
            qe = self.sens.qe
        else:
            qe = SpectralElement(ConstFlux1D, amplitude=float(self.sens_vars['qe'][2].get()))
        
        # Get nonlinearity scaleup
        if self.sens_vars['nonlinearity_scaleup'][2].get() == 'ARRAY':
            nonlinearity_scaleup = self.sens.nonlinearity_scaleup
        else:
            nonlinearity_scaleup = None
        
        # Create sensor
        sens = Sensor(
            pix_size=self.sens_vars['pix_size'][2].get(),
            read_noise=self.sens_vars['read_noise'][2].get(),
            dark_current=self.sens_vars['dark_current'][2].get(),
            full_well=self.sens_vars['full_well'][2].get(),
            qe=qe,
            nonlinearity_scaleup=nonlinearity_scaleup
        )
        
        # Create telescope
        tele = Telescope(
            diam=self.tele_vars['diam'][2].get(),
            f_num=self.tele_vars['f_num'][2].get(),
            bandpass=self.tele_vars['bandpass'][2].get()
        )
        
        # Get observing parameters
        exposure_time = self.obs_vars_dict['exptime'][2].get()
        num_exposures = int(self.obs_vars_dict['num_exposures'][2].get())
        limiting_snr = self.obs_vars_dict['limiting_snr'][2].get()
        
        # Create filter bandpass with instrument throughput
        filter_bp = filter_dict_lightspeed[self.obs_vars_dict['filter'][2].get()]
        instrument_thru_bp = self.get_instrument_throughput_bp()
        if instrument_thru_bp is not None:
            total_filter_bp = filter_bp * instrument_thru_bp
        else:
            total_filter_bp = filter_bp
        
        # Get sky conditions
        seeing_arcsec = self.obs_vars_dict['seeing'][2].get()
        obs_zo = self.obs_vars_dict['zo'][2].get()
        obs_altitude = self.tele_vars['altitude'][2].get()
        obs_alpha = self.obs_vars_dict['alpha'][2].get()
        obs_rho = self.obs_vars_dict['rho'][2].get()
        
        # Get aperture radius
        obs_aper_rad = self.obs_vars_dict['aper_rad'][2].get()
        obs_aper_rad = None if str(obs_aper_rad) in ['', 'None'] else obs_aper_rad
        
        # Create observatory
        observatory = GroundObservatory(
            sens, tele,
            exposure_time=exposure_time,
            num_exposures=num_exposures,
            limiting_s_n=limiting_snr,
            filter_bandpass=total_filter_bp,
            seeing=seeing_arcsec,
            zo=obs_zo,
            rho=obs_rho,
            altitude=obs_altitude,
            alpha=obs_alpha,
            aper_radius=obs_aper_rad
        )
        
        return observatory
    
    def set_spectrum(self):
        '''Set the spectrum based on the selected spectrum type'''
        if self.flat_spec_bool.get():
            abmag = self.flat_spec_mag.get()
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
        '''Run calculations for basic observing properties'''
        try:
            observatory = self.set_obs()
            
            # Calculate all results
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
            
            # Display results
            for key, value in self.basic_results.items():
                if key in ['lambda_pivot', 'eff_area_pivot']:
                    value[3].config(text=f'{value[2]:5.0f}')
                else:
                    value[3].config(text=f'{value[2]:4.3f}')
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)
    
    def run_observation(self):
        '''Run calculations for observing a spectrum'''
        try:
            spectrum = self.set_spectrum()
            observatory = self.set_obs()
            results = observatory.observe(spectrum)
            
            signal = results['signal']
            noise = results['tot_noise']
            snr = signal / noise
            phot_prec = 10 ** 6 / snr
            
            self.spec_results_data[0].config(text=f'{signal:3.2f}')
            self.spec_results_data[1].config(text=f'{noise:3.2f}')
            self.spec_results_data[2].config(text=f'{snr:4.3f}')
            self.spec_results_data[3].config(text=f'{phot_prec:4.3f}')
            self.spec_results_data[4].config(text=f'{results["n_aper"]:2d}')
            
            noise_str = (f'Shot noise: {results["shot_noise"]:.2f}\n'
                        f'Dark noise: {results["dark_noise"]:.2f}\n'
                        f'Read noise: {results["read_noise"]:.2f}\n'
                        f'Background noise: {results["bkg_noise"]:.2f}\n'
                        f'Scintillation noise: {results["scint_noise"]:.2f}')
            self.spec_results_data[5].config(text=noise_str)
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)
    
    def plot_trans(self):
        '''Plot the transmission of the instrument'''
        obs = self.set_obs()
        
        sens_bp = obs.sensor.qe
        tele_bp = obs.telescope.bandpass
        filter_bp = filter_dict_lightspeed[self.obs_vars_dict['filter'][2].get()]
        instrument_thru_bp = self.get_instrument_throughput_bp()
        
        atmo_throughput_with_airmass = atmo_bandpass(atmo_bandpass.waveset) ** obs.airmass
        atmo_bp = SpectralElement(Empirical1D, points=atmo_bandpass.waveset,
                                 lookup_table=atmo_throughput_with_airmass)
        
        total_bp = obs.bandpass
        
        # Build list of bandpasses to plot (skip None entries)
        bps_to_plot = [sens_bp, filter_bp, tele_bp, atmo_bp, total_bp]
        bp_names = ['Sensor QE', 'Filter', 'Telescope', 'Atmosphere', 'Total']
        linestyles = ['--', ':', ':', '-.', '-']
        
        # Add instrument throughput if it exists
        if instrument_thru_bp is not None:
            bps_to_plot.insert(2, instrument_thru_bp)
            bp_names.insert(2, 'Instrument Optics')
            linestyles.insert(2, '--')
        
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
        '''Plot the photometric precision vs. AB magnitude'''
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
        
        plt.plot(mag_points, ppm_points, label='Total Noise')
        plt.plot(mag_points, ppm_points_source, label='Shot Noise')
        plt.plot(mag_points, ppm_points_read, label='Read Noise')
        plt.plot(mag_points, ppm_points_bkg, label='Background Noise')
        plt.plot(mag_points, ppm_points_dc, label='Dark Current Noise')
        plt.plot(mag_points, ppm_points_scint, label='Scintillation Noise')
        
        ppm_threshold = 1e6 / self.obs_vars_dict['limiting_snr'][2].get()
        plt.fill_between(np.linspace(10, 30, 10), ppm_threshold, 2e6,
                        color='red', alpha=0.1, label='Non-detection')
        
        plt.xlim(10, 28)
        plt.ylim(1, 2e6)
        plt.xlabel('AB Magnitude')
        plt.ylabel('Photometric Precision (ppm)')
        plt.yscale('log')
        plt.legend()
        
        tele_name = self.tele_vars['name'][2].get()
        bandpass = self.obs_vars_dict['filter'][2].get()
        exposure_time = self.obs_vars_dict['exptime'][2].get()
        plt.title(f'{tele_name}, {bandpass}, t_exp={exposure_time}s')
        plt.show()
    
    def plot_mag_vs_snr(self):
        '''Plot the SNR vs. AB magnitude for different exposure rates'''
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.75)
        
        exp_time_vals = [0.0002, 0.001, 0.004]
        line_styles = ['k-', 'k--', 'k-.']
        labels = ['5000 Hz', '1000 Hz', '250 Hz']
        
        for exp_time, ls, label in zip(exp_time_vals, line_styles, labels):
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
            
            plt.plot(mag_points, snr_points, ls, label=label)
        
        # Add reference lines for known pulsars
        pulsar_data = [
            (16.5, 'magenta', '-.', 'SAXJ1808-3658'),
            (16.6, 'purple', ':', 'Crab'),
            (17.5, 'blue', '-.', 'J1023+0038'),
            (22.1, 'lime', ':', 'B1509-58'),
            (22.5, 'darkgreen', ':', 'B0540-69'),
            (23.6, 'orange', ':', 'Vela'),
            (25.0, 'red', ':', 'B0656+14'),
            (25.5, 'brown', ':', 'Geminga'),
            (26.0, 'cyan', '-.', 'J2339-0533')
        ]
        
        for mag, color, ls, name in pulsar_data:
            plt.axvline(x=mag, color=color, ls=ls, label=name)
        
        # Add horizontal reference lines
        for snr_val in [10, 100, 1000, 10000]:
            plt.axhline(y=snr_val, color='gray', ls='-', alpha=0.5)
        
        plt.xlim(15, 30)
        plt.ylim(1, 1e5)
        plt.xlabel('AB Magnitude')
        plt.ylabel('White light SNR in 3 hours')
        plt.yscale('log')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        plt.show()


MyGUI()