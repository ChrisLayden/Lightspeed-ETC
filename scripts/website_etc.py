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
    sensor_dict_web, 
    telescope_dict_web, 
    filter_dict_web,
    atmo_bandpass,
    lightspeed_thru,
    throughput_proto
)

app = FastAPI(title="Lightspeed ETC API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "https://yourusername.github.io",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument throughput SpectralElements (None means no additional throughput)
INSTRUMENT_THROUGHPUT = {
    'Clay (proto-Lightspeed)': throughput_proto,
    'Clay (full Lightspeed)': lightspeed_thru,
    'Clay Prime Focus': None,
    'WINTER': None,
}

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
    instrument_throughput: str  # Can be "ARRAY" or a float string
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
    instrument_throughput: str
    zo: float
    altitude: float

class MagPrecisionPlotRequest(GeneralCalcRequest):
    pass

# ==================== Helper Functions ====================

def get_instrument_throughput_for_telescope(telescope_name: str):
    """Get the instrument throughput SpectralElement for a telescope, or None"""
    for key, thru in INSTRUMENT_THROUGHPUT.items():
        if key in telescope_name:
            return thru
    return None

def get_instrument_throughput_bp(telescope_name: str, throughput_str: str):
    """Get the instrument throughput as a SpectralElement, or None if unity"""
    if throughput_str == 'ARRAY':
        return get_instrument_throughput_for_telescope(telescope_name)
    else:
        try:
            scalar_val = float(throughput_str)
        except ValueError:
            scalar_val = 1.0
        if scalar_val == 1.0:
            return None
        return SpectralElement(ConstFlux1D, amplitude=scalar_val)

def create_observatory(request: GeneralCalcRequest) -> GroundObservatory:
    """Create a GroundObservatory object from request parameters"""
    
    # Get base sensor and telescope from dictionaries
    base_sensor = sensor_dict_web[request.sensor_name]
    base_telescope = telescope_dict_web[request.telescope_name]
    
    # Create Sensor with user-specified parameters
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
    
    # Get filter bandpass with instrument throughput
    filter_bp = filter_dict_web[request.filter]
    instrument_thru_bp = get_instrument_throughput_bp(
        request.telescope_name, request.instrument_throughput)
    
    if instrument_thru_bp is not None:
        total_filter_bp = filter_bp * instrument_thru_bp
    else:
        total_filter_bp = filter_bp
    
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
        abmag = params['ab_mag']
        fluxdensity_jy = 10 ** (-0.4 * (abmag - 8.90))
        spectrum = SourceSpectrum(ConstFlux1D, amplitude=fluxdensity_jy * u.Jy)
        
    elif spectrum_type == 'blackbody':
        temp = params['temperature']
        distance = params['distance']
        l_bol = params['luminosity']
        spectrum = blackbody_spec(temp, distance, l_bol)
        
    elif spectrum_type == 'user':
        spectrum_name = params['name']
        try:
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
    return {"sensors": list(sensor_dict_web.keys())}

@app.get("/get_available_telescopes")
async def get_available_telescopes():
    """Get list of available telescopes"""
    return {"telescopes": list(telescope_dict_web.keys())}

@app.get("/get_available_filters")
async def get_available_filters():
    """Get list of available filters"""
    return {"filters": list(filter_dict_web.keys())}

@app.get("/get_sensor_info/{sensor_name}")
async def get_sensor_info(sensor_name: str):
    """Get sensor properties by name"""
    try:
        sensor = sensor_dict_web[sensor_name]
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
        telescope = telescope_dict_web[telescope_name]
        
        altitude_dict = {
            'Clay': 2516,      # Las Campanas Observatory, Chile
            'Swope': 2516,     # Las Campanas Observatory, Chile  
            'Palomar': 1712,   # Palomar Observatory, California
        }
        
        altitude = 2516  # Default to Clay/Las Campanas
        if 'Clay' in telescope_name:
            altitude = altitude_dict['Clay']
        elif 'WINTER' in telescope_name:
            altitude = altitude_dict['Palomar']
        elif 'Swope' in telescope_name:
            altitude = altitude_dict['Swope']
        
        if hasattr(telescope.bandpass, 'model') and hasattr(telescope.bandpass.model, 'amplitude'):
            bandpass_val = np.round(telescope.bandpass.model.amplitude.value, 3)
        else:
            bandpass_val = 1.0
        
        # Check if this telescope has array-based instrument throughput
        has_array_throughput = get_instrument_throughput_for_telescope(telescope_name) is not None
        
        return {
            "diam": telescope.diam,
            "f_num": telescope.f_num,
            "bandpass": bandpass_val,
            "altitude": altitude,
            "instrument_throughput": "ARRAY" if has_array_throughput else "1.0",
            "has_array_throughput": has_array_throughput
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Telescope '{telescope_name}' not found")

@app.post("/calculate_general", response_model=GeneralCalcResponse)
async def calculate_general(request: GeneralCalcRequest):
    """Calculate general observing properties"""
    try:
        observatory = create_observatory(request)
        
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
        base_sensor = sensor_dict_web[request.sensor_name]
        base_telescope = telescope_dict_web[request.telescope_name]
        
        sens_bp = base_sensor.qe
        tele_bp = base_telescope.bandpass
        filter_bp = filter_dict_web[request.filter]
        instrument_thru_bp = get_instrument_throughput_bp(
            request.telescope_name, request.instrument_throughput)
        
        # Create observatory to get airmass and total bandpass
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
        
        if instrument_thru_bp is not None:
            total_filter_bp = filter_bp * instrument_thru_bp
        else:
            total_filter_bp = filter_bp
        
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
        
        atmo_throughput_with_airmass = atmo_bandpass(atmo_bandpass.waveset) ** observatory.airmass
        atmo_bp = SpectralElement(Empirical1D, points=atmo_bandpass.waveset,
                                lookup_table=atmo_throughput_with_airmass)
        
        total_bp = observatory.bandpass
        
        # Build list of bandpasses to plot
        bps_to_plot = [sens_bp, filter_bp, tele_bp, atmo_bp, total_bp]
        bp_names = ['Sensor QE', 'Filter', 'Telescope', 'Atmosphere', 'Total']
        linestyles = ['--', ':', ':', '-.', '-']
        
        # Add instrument throughput if it exists
        if instrument_thru_bp is not None:
            bps_to_plot.insert(2, instrument_thru_bp)
            bp_names.insert(2, 'Instrument Optics')
            linestyles.insert(2, '--')
        
        traces = []
        for bp, name, ls in zip(bps_to_plot, bp_names, linestyles):
            if hasattr(bp, 'model') and hasattr(bp.model, 'amplitude') and bp.waveset is None:
                wave = np.linspace(250, 1100, 100)
                throughput = np.ones_like(wave) * bp.model.amplitude.value
                traces.append({
                    'wavelength': wave.tolist(),
                    'transmission': (throughput * 100).tolist(),
                    'name': name,
                    'linestyle': ls
                })
            else:
                wave_nm = bp.waveset.to(u.nm).value
                throughput = bp(bp.waveset).value
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
    uvicorn.run(app, host="0.0.0.0", port=8080)