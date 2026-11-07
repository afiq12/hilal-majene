# hilal-majene

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24025239.svg)](https://doi.org/10.5281/zenodo.24025239)

Reanalysis-driven atmospheric extinction and the physical visibility of the lunar crescent at a tropical coastal site.
Code and data accompanying:

> Agung, A., Arsyad, A. A., Marwan, Putriani, S. *Reanalysis-driven atmospheric extinction and the physical visibility of
> the lunar crescent at a tropical coastal site: an open pipeline and a test of the MABIMS criterion at Majene, Indonesia.*

The pipeline couples a JPL ephemeris (Skyfield) to a Schaefer-type twilight-sky and crescent-contrast model, restated in SI
photometric units and closed with the contrast-threshold model of Crumey (2014, MNRAS 442, 2600). The visual extinction
coefficient is computed evening by evening from NASA POWER reanalysis meteorology (MERRA-2 humidity and temperature,
CERES aerosol optical depth and cloud amount). The model has no tuned psychophysical parameter.

## Contents

| Path | Content |
| --- | --- |
| `ephemeris.py` | Site definition, conjunctions, sunset and moonset, crescent geometry (altitudes, elongation, ARCV, DAZ, width W', lag), Yallop best time |
| `atmosphere.py` | NASA POWER / ERA5 CSV reader; V-band extinction (Rayleigh, aerosol from AOD or humidity, ozone, water vapour); Rozenberg air mass |
| `contrast.py` | Twilight sky (Sultan 2007 anchor, Schaefer 1998 alternative), crescent luminance (Krisciunas & Schaefer 1991), Crumey (2014) threshold, physical visibility index; MABIMS, Yallop q, Odeh V |
| `run_majene.py` | Full pipeline for one site, 2000 to 2030 |
| `run_grid.py` | Same pipeline for the eight West Sulawesi points in `data/grid/` |
| `calibrate.py` | Calibration offset for three reference extinctions, critical extinction, classification tables |
| `make_figures*.py`, `make_map_sulbar.py` | Figures 2 to 8 of the paper (`LANG_FIG=en` for English labels) |
| `data/` | NASA POWER daily extracts (Majene and eight grid points), seven public rukyat reports of Kanwil Kemenag Sulawesi Barat (2023 to 2026) |
| `results/` | Per-evening tables (384 evenings) for Majene and the grid points, calibrated table, grid summary |
| `figures/` | Figures as used in the paper |

Code comments are partly in Indonesian; all function names, outputs and this documentation are in English.

## Installation and use

```bash
pip install -r requirements.txt
python run_majene.py --start 2000 --end 2030 --met data/power_majene_daily.csv      # about 1 min
python calibrate.py output/hilal_majene_2000_2030.csv
python run_grid.py                                                                    # about 9 min
LANG_FIG=en python make_figures.py && LANG_FIG=en python make_figures_calib.py output/hilal_majene_2000_2030_calibrated.csv
```

Options of `run_majene.py`: `--threshold crumey|schaefer` (default crumey), `--sky-model sultan07|schaefer98` (default sultan07),
`--sky-scale`, `--field-factor`, `--sb-factor`, `--lat --lon --elev` for another site.
The ephemeris defaults to DE421 shipped with `skyfield-data`; place `de440s.bsp` in `data/` to use DE440s.

## Data sources and licences

NASA POWER data are provided by the NASA Langley Research Center POWER Project funded through the NASA Earth Science
Directorate Applied Science Program (https://power.larc.nasa.gov). Rukyat reports are transcribed from the public news
pages of Kanwil Kementerian Agama Sulawesi Barat (URLs in the CSV). Code is released under the MIT licence (`LICENSE`).

## How to cite

Please cite the paper and the archived version of this repository (see `CITATION.cff` and the Zenodo DOI badge).
