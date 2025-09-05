from synphot import SourceSpectrum, Empirical1D
from synphot.models import BlackBody1D, BlackBodyNorm1D
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
            The bolometric luminosity of the source, in erg/s.
        '''
    # Create a blackbody spectrum. Default spectrum is normalized to
    # a radius of 1 R_SUN at a distance of 1 kpc.
    spectrum = SourceSpectrum(BlackBodyNorm1D, temperature=temp * u.K)
    wavelengths = spectrum.waveset
    flux_values = spectrum(wavelengths)
    ztab = RedshiftLookup()
    initial_z = ztab(10 ** -3)
    obs_z = ztab(dist)
    
    # Adjust the wavelengths of the source spectrum to account for
    # the redshift, and the flux for the luminosity distance.
    obs_wave = wavelengths * (1+initial_z) / (1+obs_z)
    obs_flux = (flux_values * (1+initial_z) / (1+obs_z))
    
    # Scale the flux using the desired bolometric luminosity
    l_bol_scaling = l_bol / (4 * np.pi * constants.sigma *
                             constants.R_SUN ** 2 * temp ** 4)
    l_bol_scaling = l_bol_scaling / (dist * 1000) ** 2
    obs_flux *= l_bol_scaling
    
    # Create the observed spectrum
    obs_spectrum = SourceSpectrum(Empirical1D, points=obs_wave,
                                lookup_table=obs_flux)
    return obs_spectrum

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    spec = SourceSpectrum(BlackBodyNorm1D, temperature=5777)
    print(spec(5000*u.AA, flux_unit=FLAM) * (2 * 10 ** 8) ** 2)
