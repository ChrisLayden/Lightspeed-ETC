# Lightspeed-ETC
An exposure time calculator for the Lightspeed instrument planned for installation on the Magellan Clay Telescope.

Requires synphot, matplotlib, astropy, numpy, and scipy.

CPL 04/23/2025

How to use:
1) Install synphot: use

```
pip install synphot
```

If this doesn't work, see more information at https://synphot.readthedocs.io/en/latest/

2) To open the Lightspeed ETC, use

```
cd scripts
python lightspeed_etc.py
```

There you can choose observing parameters and see Lightspeed's response for a given source.
