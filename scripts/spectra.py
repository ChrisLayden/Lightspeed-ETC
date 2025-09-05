from synphot import SourceSpectrum, Empirical1D
from synphot.models import BlackBody1D, PowerLawFlux1D
import astropy.units as u
from synphot.units import FLAM
from redshift_lookup import RedshiftLookup
import numpy as np
import constants


def blackbody_spec(temp, dist, l_bol):
    '''Returns a blackbody spectrum with the desired properties.

        Parameters
        ----------
        temp: float
            The temperature of the blackbody, in K.
        dist: float
            The luminosity distance at which the spectrum is
            specified, in Mpc.
        l_bol: float
            The bolometric luminosity of the source.
        '''
    # Create a blackbody spectrum
    # In synphot, we need to specify temperature with units
    spectrum = SourceSpectrum(BlackBody1D, temperature=temp * u.K)
    
    ztab = RedshiftLookup()
    initial_z = ztab(10 ** -3)
    obs_z = ztab(dist)
    
    # Get wavelengths and flux from the spectrum
    # Use a reasonable wavelength range for evaluation
    wavelengths = spectrum.waveset
    if wavelengths is None or len(wavelengths) < 3:
        # If no waveset defined, create a reasonable range
        wavelengths = np.arange(100, 100000, 10) * u.AA
    
    flux_values = spectrum(wavelengths)
    
    # Adjust the wavelengths of the source spectrum to account for
    # the redshift, and the flux for the luminosity distance.
    obs_wave = wavelengths * (1+initial_z) / (1+obs_z)
    obs_flux = (flux_values * (1+initial_z) / (1+obs_z)
                * (10 ** -3 / dist) ** 2)
    
    # Scale the flux using the desired bolometric luminosity
    l_bol_scaling = l_bol / (4 * np.pi * constants.sigma *
                             constants.R_SUN ** 2 * temp ** 4)
    obs_flux *= l_bol_scaling
    
    # Create the observed spectrum
    obs_spectrum = SourceSpectrum(Empirical1D, points=obs_wave,
                                lookup_table=obs_flux)
    return obs_spectrum
