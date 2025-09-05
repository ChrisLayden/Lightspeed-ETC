'''Subclass of Observatory for ground-based telescopes.

Classes
-------
GroundObservatory
    A class for ground-based observatories, inheriting from Observatory.
    Key differences are the addition of scintillation noise (using airmass
    and altitude values), atmospheric seeing, and the effects of the moon
    on the sky background (ZZZ that's not yet added).
'''


import os
import numpy as np
from synphot import SpectralElement, SourceSpectrum, ConstFlux1D, Empirical1D
import astropy.units as u
from synphot.units import FLAM
from observatory import Observatory, Sensor, Telescope
from instruments import atmo_bandpass
from sky_background import bkg_spectrum_ground
import psfs

class GroundObservatory(Observatory):
    '''A class for ground-based observatories, inheriting from Observatory.'''

    def __init__(self, sensor, telescope, filter_bandpass=SpectralElement(ConstFlux1D, amplitude=1.0),
                 exposure_time=1., num_exposures=1, seeing=1.0,
                 limiting_s_n=5., altitude=0, alpha=180, zo=0, rho=45,
                 aper_radius=None):
        '''Initialize the GroundObservatory class.
        
        Parameters
        ----------
        sensor: Sensor
            The sensor object to be used for observations.
        telescope: Telescope
            The telescope object to be used for observations.
        filter_bandpass: synphot.SpectralElement object
            The bandpass filter to be used for observations.
        exposure_time: float
            The exposure time for observations, in seconds.
        num_exposures: int
            The number of exposures to be taken.
        seeing: float
            The PSF FWHM, in arcseconds. Assumes a Gaussian PSF, and that
            all broadening is due to the atmosphere.
        limiting_s_n: float
            The limiting signal-to-noise ratio for observations.
        altitude: float
            The altitude of the observatory, in meters.
        alpha: float
            The lunar phase angle, in degrees. 0 is full moon, 180 is new moon.
            Should only be between 0 and 180 (assumes symmetry between
            waning/waxing).
        zo: float
            The zenith angle of the object being observed, in degrees.
        rho: float
            The angular separation between the moon and the object, in degrees.
            For simplicity, we assume the moon is lower in the sky than the
            object, with zenith angle zo - rho.
        aper_rad: int
            The radius of the aperture, in pixels. If None, the optimal
            aperture will be calculated.
        '''

        super().__init__(sensor=sensor, telescope=telescope,
                         filter_bandpass=filter_bandpass,
                         exposure_time=exposure_time,
                         num_exposures=num_exposures,
                         limiting_s_n=limiting_s_n,
                         aper_radius=aper_radius)

        telescope.psf_type = 'gaussian'
        telescope.fwhm = seeing
        self.alpha = alpha
        self.altitude = altitude
        self.zo = zo
        # Formula 3 in Krisciunas & Schaefer 1991 for airmass.
        self.airmass = (1 - 0.96 * np.sin(np.radians(zo)) ** 2) ** -0.5
        # atmospheric transmission from https://arxiv.org/pdf/0708.1364
        atmo_throughput_with_airmass = atmo_bandpass(atmo_bandpass.waveset) ** self.airmass
        atmo_bp = SpectralElement(Empirical1D, points=atmo_bandpass.waveset,
                                lookup_table=atmo_throughput_with_airmass)
        self.bandpass = (filter_bandpass * self.telescope.bandpass *
                         self.sensor.qe * atmo_bp)
        self.eff_area = self.bandpass * SpectralElement(ConstFlux1D, 
                                                       amplitude=np.pi * self.telescope.diam ** 2 / 4)
        self.lambda_pivot = self.bandpass.pivot()
        self.rho = rho
        # The zenith angle of the moon, in degrees.
        self.zm = zo - rho
        self.scint_noise = self.get_scint_noise()

    @property
    def alpha(self):
        '''The lunar phase angle during observation.'''
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0 or value > 180:
            raise ValueError("alpha must be between 0 and 180 degrees.")
        else:
            self._alpha = value

    def get_scint_noise(self):
        '''Calculate the scintillation noise, per formula from Young, 1967.'''
        diam_factor = self.telescope.diam ** - (2/3)
        exp_time_factor = (2 * self.exposure_time) ** (-1/2)
        airmass_factor = self.airmass ** (3/2)
        altitude_factor = np.exp(-self.altitude / 8000)
        return 0.09 * diam_factor * exp_time_factor * airmass_factor * altitude_factor

    def bkg_per_pix(self):
        '''The background noise per pixel, in e-/pix.'''
        bkg_wave, bkg_ilam = bkg_spectrum_ground(alpha=self.alpha, rho=self.rho,
                                                 Zm=self.zm, Zo=self.zo,)
        bkg_flam = bkg_ilam * self.pix_scale ** 2
        bkg_sp = SourceSpectrum(Empirical1D, points=bkg_wave * u.AA,
                              lookup_table=bkg_flam * FLAM)
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal

    def turnover_exp_time(self):
        '''Get the exposure time at which background noise is equal to read noise.'''
        read_bkg_ratio = self.sensor.read_noise / np.sqrt(self.bkg_per_pix())
        return self.exposure_time * read_bkg_ratio ** 2

    def get_aper(self, spectrum=None, pos=np.array([0, 0]), img_size=11,
                     resolution=11):
        '''Find the optimal aperture for a given point source.
        
        Parameters
        ----------
        spectrum: synphot.SourceSpectrum object
            The spectrum of the point source to observe.
        pos: array-like (default [0, 0])
            The centroid position of the source on the central pixel,
            in microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        '''
        if self.aper_radius is not None:
            # Make an aperture with the specified radius on the subarray,
            # centered on the central pixel.
            img_size = np.max([self.aper_radius * 2 + 3, img_size])
            aper = np.zeros((img_size, img_size))
            # For pixels within a distance of aper_radius from the center,
            # set the pixel value to 1.
            # Get an array of pixel distances from the center
            x = np.arange(img_size) - img_size // 2
            y = np.arange(img_size) - img_size // 2
            x, y = np.meshgrid(x, y)
            pix_distances = np.sqrt(x ** 2 + y ** 2)
            # Array of 1s and 0s, where 1 is within the aperture radius
            aper[pix_distances <= self.aper_radius] = 1
            return aper
        else:
            if spectrum is None:
                raise ValueError('Spectrum must be specified to find the optimal aperture.')
            aper_found = False
            while not aper_found:
                signal_grid_fine = self.signal_grid_fine(spectrum, pos, img_size, resolution)
                signal_grid = signal_grid_fine.reshape((img_size, resolution,
                                                        img_size, resolution)).sum(axis=(1, 3))
                if self.sensor.nonlinearity_scaleup is not None:
                    dark_signal = self.sensor.dark_current * self.exposure_time + self.bkg_per_pix()
                    tot_frame = signal_grid * (1 + dark_signal)
                    pix_scaleup_factors = np.interp(tot_frame,
                                                    self.sensor.nonlinearity_scaleup[:, 0],
                                                    self.sensor.nonlinearity_scaleup[:, 1],
                                                    left=self.sensor.nonlinearity_scaleup[0, 1],
                                                    right=self.sensor.nonlinearity_scaleup[-1, 1])
                    optimal_aper = psfs.get_optimal_aperture_nonlinear(signal_grid, read_noise=self.sensor.read_noise,
                                                             dark_signal=dark_signal,
                                                             pix_scaleup_factors=pix_scaleup_factors)
                else:
                    optimal_aper = psfs.get_optimal_aperture(signal_grid, self.single_pix_noise(),
                                                             scint_noise=self.scint_noise)
                # optimal_aper = psfs.get_optimal_aperture(stack_image, stack_pix_noise,
                #                                         scint_noise=stack_scint_noise)
                aper_pads = psfs.get_aper_padding(optimal_aper)
                if min(aper_pads) > 0:
                    aper_found = True
                else:
                    img_size += 5
            return optimal_aper

    def observe(self, spectrum, pos=np.array([0, 0]), img_size=11,
                resolution=11):
        '''Determine the signal and noise for observation of a point source.

        Parameters
        ----------
        spectrum: synphot.SourceSpectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels. Should be odd, so that
            [0,0] indeed represents the center of the central pixel.
            If it isn't odd, we'll just add one.
        resolution: int (default 11)
            The number of sub-pixels per pixel. Should be odd, so that
            [0,0] represents the center of the central subpixel.
            If it isn't odd, we'll just add one.

        Returns
        -------
        results_dict: dict
            A dictionary containing the signal and noise values for the
            observation. The keys are 'signal', 'tot_noise',
            'dark_noise', 'bkg_noise', 'read_noise', 'shot_noise', 'n_aper',
            and 'snr'.
        '''
        # For a realistic image with all noise sources, use the get_images method.
        # Here we just summarize signal and noise characteristics.
        optimal_aper = self.get_aper(spectrum, pos, img_size, resolution)
        img_size = optimal_aper.shape[0]
        signal_grid_fine = self.signal_grid_fine(spectrum, pos, img_size, resolution)
        signal_grid = signal_grid_fine.reshape((img_size, resolution,
                                                img_size, resolution)).sum(axis=(1, 3))
        signal = np.sum(signal_grid * optimal_aper) * self.num_exposures
        n_aper = np.sum(optimal_aper)
        if self.sensor.nonlinearity_scaleup is not None:
            # Adjust shot and read noise for sensor nonlinearity
            tot_frame = signal_grid * optimal_aper * (1 + self.sensor.dark_current * self.exposure_time +
                                self.bkg_per_pix())
            pix_scaleup_factors = np.interp(tot_frame, self.sensor.nonlinearity_scaleup[:, 0],
                                            self.sensor.nonlinearity_scaleup[:, 1],
                                            left=self.sensor.nonlinearity_scaleup[0, 1],
                                            right=self.sensor.nonlinearity_scaleup[-1, 1])
            pix_scaleup_factors = optimal_aper * pix_scaleup_factors
            pix_read_var_vals = (pix_scaleup_factors * self.sensor.read_noise) ** 2
            pix_shot_var_vals = tot_frame * pix_scaleup_factors
            pix_dark_noise_vals = self.sensor.dark_current * self.exposure_time * pix_scaleup_factors
            bkg_noise_vals = self.bkg_per_pix() * pix_scaleup_factors
            read_noise = np.sqrt(np.sum(pix_read_var_vals) * self.num_exposures)
            shot_noise = np.sqrt(np.sum(pix_shot_var_vals) * self.num_exposures)
            dark_noise = np.sqrt(np.sum(pix_dark_noise_vals) * self.num_exposures)
            bkg_noise = np.sqrt(np.sum(bkg_noise_vals) * self.num_exposures)
        else:
            shot_noise = np.sqrt(signal)
            dark_noise = np.sqrt(n_aper * self.num_exposures *
                                self.sensor.dark_current * self.exposure_time)
            bkg_noise = np.sqrt(n_aper * self.num_exposures * self.bkg_per_pix())
            read_noise = np.sqrt(n_aper * self.num_exposures * self.sensor.read_noise ** 2)
        scint_noise = signal * self.scint_noise / np.sqrt(self.num_exposures)
        tot_noise = np.sqrt(shot_noise ** 2 + dark_noise ** 2 + scint_noise ** 2 +
                            bkg_noise ** 2 + read_noise ** 2)
        results_dict = {'signal': signal, 'tot_noise': tot_noise,
                        'dark_noise': dark_noise, 'bkg_noise': bkg_noise,
                        'read_noise': read_noise, 'shot_noise': shot_noise,
                        'scint_noise': scint_noise, 'img_size': img_size,
                        'n_aper': int(n_aper), 'snr': signal / tot_noise}
        return results_dict


if __name__ == '__main__':
    from instruments import qcmos, magellan_tele_prototype, sloan_gprime
    prototype = GroundObservatory(sensor=qcmos, telescope=magellan_tele_prototype,
                                 filter_bandpass=sloan_gprime,
                                 altitude=2, exposure_time=0.1,
                                 seeing=0.5, alpha=180, zo=0, rho=45)
    my_spectrum = SourceSpectrum(ConstFlux1D, amplitude=20 * u.ABmag)
    print(prototype.observe(my_spectrum))