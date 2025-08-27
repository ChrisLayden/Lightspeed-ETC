# Lightspeed-ETC
An exposure time calculator for the Lightspeed instrument planned for installation on the Magellan Clay Telescope.

Requires tkinter, pysynphot, matplotlib, astropy, numpy, and scipy.

CPL 04/23/2025

Steps to make pysynphot operational:
1) Install pysynphot: use

```
pip install pysynphot
```

If this doesn't work, see more information at https://pysynphot.readthedocs.io/en/latest/

2) Download just the first two sets of pysynphot data files
(http://ssb.stsci.edu/trds/tarfiles/synphot1.tar.gz and
http://ssb.stsci.edu/trds/tarfiles/synphot2.tar.gz); unpack these
to some directory /my/dir

3) set
```
export PYSYN_CDBS=/my/dir/grp/redcat/trds
```

To open the Lightspeed ETC, use

```
cd scripts
python lightspeed_etc.py
```

There you can choose observing parameters and see Lightspeed's response for a given source.
