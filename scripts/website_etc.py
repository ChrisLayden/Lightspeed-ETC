"""
FastAPI Backend for Lightspeed Exposure Time Calculator
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

# Import your existing ETC modules
from synphot import SpectralElement, SourceSpectrum, ConstFlux1D, Empirical1D
import astropy.units as u
from synphot.units import FLAM
import numpy as np

from spectra import blackbody_spec
from observatory import Sensor, Telescope
from ground_observatory import GroundObservatory
from instruments import (
    sensor_dict_lightspeed, 
    telescope_dict_lightspeed, 
    filter_dict_lightspeed,
    atmo_bandpass
)

app = FastAPI(title="Lightspeed ETC API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "https://yourusername.github.io",  # Replace with your GitHub Pages URL
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request/Response Models ====================

class GeneralCalcRequest(BaseModel):
    sensor_name: str
    pix_size: float
    read_noise: float
    dark_current: float
    full_well: float
    telescope_name: str
    diameter: float
    f_number: float
    bandpass: float
    altitude: float
    exptime: float
    num_exposures: int
    filter: str
    reim_throughput: float
    limiting_snr: float
    seeing: float
    zo: float
    alpha: float
    rho: float
    aper_rad: Optional[str] = None

class SpectrumCalcRequest(GeneralCalcRequest):
    spectrum_type: str
    spectrum_params: Dict[str, Any]

class GeneralCalcResponse(BaseModel):
    pix_scale: float
    lambda_pivot: float
    psf_fwhm: float
    central_pix_frac: float
    eff_area_pivot: float
    limiting_mag: float
    saturating_mag: float
    airmass: float
    zero_point: float
    exptime_turnover: float

class SpectrumCalcResponse(BaseModel):
    signal: float
    tot_noise: float
    snr: float
    phot_prec: float
    n_aper: int
    shot_noise: float
    dark_noise: float
    read_noise: float
    bkg_noise: float
    scint_noise: float

class TransmissionPlotRequest(BaseModel):
    sensor_name: str
    telescope_name: str
    filter: str
    reim_throughput: float
    zo: float
    altitude: float

class MagPrecisionPlotRequest(GeneralCalcRequest):
    pass

# ==================== Helper Functions ====================

def create_observatory(request: GeneralCalcRequest) -> GroundObservatory:
    """Create a GroundObservatory object from request parameters"""
    
    # Get base sensor and telescope from dictionaries
    base_sensor = sensor_dict_lightspeed[request.sensor_name]
    base_telescope = telescope_dict_lightspeed[request.telescope_name]
    
    # Create Sensor with user-specified parameters
    # QE is always from the predefined sensor (ARRAY)
    sens = Sensor(
        pix_size=request.pix_size,
        read_noise=request.read_noise,
        dark_current=request.dark_current,
        full_well=request.full_well,
        qe=base_sensor.qe,
        nonlinearity_scaleup=base_sensor.nonlinearity_scaleup
    )
    
    # Create Telescope with user-specified parameters
    tele = Telescope(
        diam=request.diameter,
        f_num=request.f_number,
        bandpass=request.bandpass
    )
    
    # Get filter bandpass
    filter_bp = filter_dict_lightspeed[request.filter]
    
    # Create reimaging bandpass
    reimaging_bp = SpectralElement(ConstFlux1D, amplitude=request.reim_throughput)
    
    # Combine filter and reimaging
    total_filter_bp = filter_bp * reimaging_bp
    
    # Parse aperture radius
    aper_rad = None
    if request.aper_rad and request.aper_rad.strip():
        try:
            aper_rad = float(request.aper_rad)
        except ValueError:
            aper_rad = None
    
    # Create observatory
    observatory = GroundObservatory(
        sens, tele,
        exposure_time=request.exptime,
        num_exposures=request.num_exposures,
        limiting_s_n=request.limiting_snr,
        filter_bandpass=total_filter_bp,
        seeing=request.seeing,
        zo=request.zo,
        rho=request.rho,
        altitude=request.altitude,
        alpha=request.alpha,
        aper_radius=aper_rad
    )
    
    return observatory

def create_spectrum(spectrum_type: str, params: Dict[str, Any]) -> SourceSpectrum:
    """Create a SourceSpectrum object from spectrum parameters"""
    
    if spectrum_type == 'flat':
        # Flat spectrum at AB magnitude
        abmag = params['ab_mag']
        # Convert to Jansky
        fluxdensity_jy = 10 ** (-0.4 * (abmag - 8.90))
        spectrum = SourceSpectrum(ConstFlux1D, amplitude=fluxdensity_jy * u.Jy)
        
    elif spectrum_type == 'blackbody':
        # Blackbody spectrum
        temp = params['temperature']
        distance = params['distance']
        l_bol = params['luminosity']
        spectrum = blackbody_spec(temp, distance, l_bol)
        
    elif spectrum_type == 'user':
        # User-defined spectrum from spectra.py
        spectrum_name = params['name']
        try:
            # This assumes the spectrum is defined in the global namespace
            # or can be imported from spectra.py
            spectrum = eval(spectrum_name)
        except Exception as e:
            raise ValueError(f"Could not load spectrum '{spectrum_name}': {str(e)}")
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}")
    
    return spectrum

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {"message": "Lightspeed ETC API is running"}

@app.get("/get_available_sensors")
async def get_available_sensors():
    """Get list of available sensors"""
    return {"sensors": list(sensor_dict_lightspeed.keys())}

@app.get("/get_available_telescopes")
async def get_available_telescopes():
    """Get list of available telescopes"""
    return {"telescopes": list(telescope_dict_lightspeed.keys())}

@app.get("/get_available_filters")
async def get_available_filters():
    """Get list of available filters"""
    return {"filters": list(filter_dict_lightspeed.keys())}

@app.get("/get_sensor_info/{sensor_name}")
async def get_sensor_info(sensor_name: str):
    """Get sensor properties by name"""
    try:
        sensor = sensor_dict_lightspeed[sensor_name]
        return {
            "pix_size": sensor.pix_size,
            "read_noise": sensor.read_noise,
            "dark_current": sensor.dark_current,
            "full_well": sensor.full_well
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_name}' not found")

@app.get("/get_telescope_info/{telescope_name}")
async def get_telescope_info(telescope_name: str):
    """Get telescope properties by name"""
    try:
        telescope = telescope_dict_lightspeed[telescope_name]
        
        # Get altitude based on telescope name
        altitude_dict = {
            'Clay': 2516,      # Las Campanas Observatory, Chile
            'Swope': 2516,     # Las Campanas Observatory, Chile  
            'Palomar': 1712,   # Palomar Observatory, California
            'Keck': 4145,      # Mauna Kea, Hawaii
            'GTC': 2396,       # Roque de los Muchachos, La Palma
            'GMT': 2516        # GMT will also be at Las Campanas
        }
        
        altitude = 2516  # Default to Clay/Las Campanas
        if 'Clay' in telescope_name or 'GMT' in telescope_name:
            altitude = altitude_dict['Clay']
        elif 'WINTER' in telescope_name or 'Hale' in telescope_name:
            altitude = altitude_dict['Palomar']
        elif 'Keck' in telescope_name:
            altitude = altitude_dict['Keck']
        elif 'GTC' in telescope_name:
            altitude = altitude_dict['GTC']
        elif 'Swope' in telescope_name:
            altitude = altitude_dict['Swope']
        
        # Get bandpass value
        if hasattr(telescope.bandpass, 'model') and hasattr(telescope.bandpass.model, 'amplitude'):
            bandpass_val = np.round(telescope.bandpass.model.amplitude.value, 3)
        else:
            bandpass_val = 1.0
        
        return {
            "diam": telescope.diam,
            "f_num": telescope.f_num,
            "bandpass": bandpass_val,
            "altitude": altitude
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Telescope '{telescope_name}' not found")

@app.post("/get_reimaging_throughput")
async def get_reimaging_throughput(data: Dict[str, str]):
    """Get reimaging throughput based on telescope and filter"""
    telescope_name = data.get('telescope', '')
    filter_name = data.get('filter', '')
    
    throughput_dict_prototype = {
        "Baader g'": 0.57, "Baader r'": 0.65,
        "Baader i'": 0.28, "Baader z'": 0.06,
        "Baader u'": 0.05, "Halpha": 0.65,
        "Baader OIII": 0.57
    }
    
    throughput_dict_lightspeed = {
        "Baader g'": 0.8, "Baader r'": 0.8,
        "Baader i'": 0.8, "Baader z'": 0.8,
        "Baader u'": 0.8, "Halpha": 0.8, 
        "None": 0.8, "Baader OIII": 0.8
    }
    
    throughput_dict_prime = {
        "Baader g'": 1.0, "Baader r'": 1.0,
        "Baader i'": 1.0, "Baader z'": 1.0,
        "Baader u'": 1.0, "Halpha": 1.0, 
        "None": 1.0, "Baader OIII": 1.0
    }
    
    if telescope_name == 'Clay (proto-Lightspeed)':
        throughput_dict = throughput_dict_prototype
    elif telescope_name == 'Clay (full Lightspeed)' or 'Keck' in telescope_name:
        throughput_dict = throughput_dict_lightspeed
    elif telescope_name == 'Clay Prime Focus':
        throughput_dict = throughput_dict_prime
    else:
        return {"throughput": 1.0}
    
    throughput = throughput_dict.get(filter_name, 1.0)
    return {"throughput": throughput}

@app.post("/calculate_general", response_model=GeneralCalcResponse)
async def calculate_general(request: GeneralCalcRequest):
    """Calculate general observing properties"""
    try:
        observatory = create_observatory(request)
        
        # Calculate PSF FWHM in arcseconds
        fwhm_arcsec = (observatory.psf_fwhm_um() * observatory.pix_scale /
                       observatory.sensor.pix_size)
        
        results = {
            "pix_scale": observatory.pix_scale,
            "lambda_pivot": observatory.lambda_pivot.to(u.AA).value,
            "psf_fwhm": fwhm_arcsec,
            "central_pix_frac": observatory.central_pix_frac() * 100,
            "eff_area_pivot": observatory.eff_area(observatory.lambda_pivot).value,
            "limiting_mag": observatory.limiting_mag(),
            "saturating_mag": observatory.saturating_mag(),
            "airmass": observatory.airmass,
            "zero_point": observatory.zero_point_mag(),
            "exptime_turnover": observatory.turnover_exp_time()
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_spectrum", response_model=SpectrumCalcResponse)
async def calculate_spectrum(request: SpectrumCalcRequest):
    """Calculate spectrum observation results"""
    try:
        observatory = create_observatory(request)
        spectrum = create_spectrum(request.spectrum_type, request.spectrum_params)
        
        # Run observation
        results = observatory.observe(spectrum)
        
        signal = results['signal']
        noise = results['tot_noise']
        snr = signal / noise
        phot_prec = 10 ** 6 / snr
        
        response = {
            "signal": signal,
            "tot_noise": noise,
            "snr": snr,
            "phot_prec": phot_prec,
            "n_aper": results['n_aper'],
            "shot_noise": results['shot_noise'],
            "dark_noise": results['dark_noise'],
            "read_noise": results['read_noise'],
            "bkg_noise": results['bkg_noise'],
            "scint_noise": results['scint_noise']
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot_transmission")
async def plot_transmission(request: TransmissionPlotRequest):
    """Generate transmission plot data for all components"""
    try:
        # Get components
        base_sensor = sensor_dict_lightspeed[request.sensor_name]
        base_telescope = telescope_dict_lightspeed[request.telescope_name]
        
        sens_bp = base_sensor.qe
        tele_bp = base_telescope.bandpass
        filter_bp = filter_dict_lightspeed[request.filter]
        reimaging_bp = SpectralElement(ConstFlux1D, amplitude=request.reim_throughput)
        
        # Create observatory to get airmass
        sens = Sensor(
            pix_size=base_sensor.pix_size,
            read_noise=base_sensor.read_noise,
            dark_current=base_sensor.dark_current,
            full_well=base_sensor.full_well,
            qe=base_sensor.qe,
            nonlinearity_scaleup=base_sensor.nonlinearity_scaleup
        )
        tele = Telescope(
            diam=base_telescope.diam,
            f_num=base_telescope.f_num,
            bandpass=base_telescope.bandpass
        )
        total_filter_bp = filter_bp * reimaging_bp
        observatory = GroundObservatory(
            sens, tele,
            exposure_time=1.0,
            num_exposures=1,
            limiting_s_n=5.0,
            filter_bandpass=total_filter_bp,
            seeing=0.5,
            zo=request.zo,
            rho=45,
            altitude=request.altitude,
            alpha=180,
            aper_radius=None
        )
        
        # Get atmosphere with airmass
        atmo_throughput_with_airmass = atmo_bandpass(atmo_bandpass.waveset) ** observatory.airmass
        atmo_bp = SpectralElement(Empirical1D, points=atmo_bandpass.waveset,
                                lookup_table=atmo_throughput_with_airmass)
        
        total_bp = observatory.bandpass
        
        # Prepare traces
        traces = []
        bps_to_plot = [sens_bp, filter_bp, reimaging_bp, tele_bp, atmo_bp, total_bp]
        bp_names = ['Sensor QE', 'Filter', 'Reimaging Optics', 'Telescope', 'Atmosphere', 'Total']
        linestyles = ['--', ':', '--', ':', '-.', '-']
        
        for bp, name, ls in zip(bps_to_plot, bp_names, linestyles):
            if hasattr(bp, 'model') and hasattr(bp.model, 'amplitude') and bp.waveset is None:
                # Uniform transmission
                wave = np.linspace(250, 1100, 100)
                throughput = np.ones_like(wave) * bp.model.amplitude.value
                traces.append({
                    'wavelength': wave.tolist(),
                    'transmission': (throughput * 100).tolist(),
                    'name': name,
                    'linestyle': ls
                })
            else:
                # Array-based bandpass
                wave_nm = bp.waveset.to(u.nm).value
                throughput = bp(bp.waveset).value  # Add .value to convert from Quantity
                traces.append({
                    'wavelength': wave_nm.tolist(),
                    'transmission': (throughput * 100).tolist(),
                    'name': name,
                    'linestyle': ls
                })
        
        return {'traces': traces}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot_mag_vs_precision")
async def plot_mag_vs_precision(request: MagPrecisionPlotRequest):
    """Generate magnitude vs photometric precision plot data"""
    try:
        observatory = create_observatory(request)
        
        # Calculate for range of magnitudes
        mag_points = np.linspace(10, 28, 15)
        ppm_points = np.zeros_like(mag_points)
        ppm_points_source = np.zeros_like(mag_points)
        ppm_points_read = np.zeros_like(mag_points)
        ppm_points_bkg = np.zeros_like(mag_points)
        ppm_points_dc = np.zeros_like(mag_points)
        ppm_points_scint = np.zeros_like(mag_points)
        
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
        
        ppm_threshold = 1e6 / request.limiting_snr
        
        # Create title
        title = f"{request.telescope_name}, {request.filter}, t_exp={request.exptime}s"
        
        return {
            'magnitudes': mag_points.tolist(),
            'total_noise': ppm_points.tolist(),
            'shot_noise': ppm_points_source.tolist(),
            'read_noise': ppm_points_read.tolist(),
            'bkg_noise': ppm_points_bkg.tolist(),
            'dark_noise': ppm_points_dc.tolist(),
            'scint_noise': ppm_points_scint.tolist(),
            'limiting_threshold': ppm_threshold,
            'title': title
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Main ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)