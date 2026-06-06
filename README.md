# Impacts-of-Donor-Catchment-Number-on-Data-Driven-Streamflow-Prediction-in-Ungauged-Basins
This repository is intended to provide code for data analysis of literature.
# Code and Data Package

This folder contains the curated code, input data, shapefile, and output figures used for the analysis of geographic and similarity-based watershed comparison results.

All scripts have been updated to use relative paths. Input CSV files are stored in `data/csv`, the shapefile is stored in `data/shp`, source scripts are stored in `code`, and exported figures are stored in `output`.

## Folder Structure

```text
code/
├─ code/                      Python scripts
├─ data/
│  ├─ csv/                    Input CSV files used by the scripts
│  └─ shp/                    U.S. boundary shapefile and sidecar files
├─ output/                    Exported result figures
└─ README.md
```

## Contents

### Scripts

The `code/` subfolder contains 12 Python scripts:

- `2-Geographical location.py`
- `3-method compare.py`
- `4-KGE Variance distribution.py`
- `4-median distribution.py`
- `5-boxplot.py`
- `6-geo-best kge.py`
- `6-similarity-best kge.py`
- `7-point_density_heatmap.py`
- `7-similarity point density heatmap.py`
- `8-geo density.py`
- `8-similarity density.py`
- `9-spearman.py`

### Input Data

The `data/csv/` folder contains the CSV files required by the scripts, including:

- KGE result tables for geographic and similarity methods at multiple `N` values
- merged attribute and KGE summary tables
- station attribute tables
- density analysis input table

The `data/shp/` folder contains the U.S. boundary shapefile:

- `cb_2018_us_state_20m.shp`
- associated sidecar files (`.dbf`, `.shx`, `.prj`, `.cpg`, etc.)

### Output Figures

The `output/` folder currently contains:

- `figure1.tif`
- `Figure2.tif`
- `Figure3.tif`
- `Figure4.tif`
- `Figure5.tif`
- `Figure6-1.tif`
- `Figure6-2.tif`
- `Figure7.tif`
- `Figure8.tif`
- `Figure9.tif`

## Data Notes

Two versions of `filtered_camels_10day_attributes_cleaned` are included because they were used in different scripts:

- `filtered_camels_10day_attributes_cleaned_learn.csv`
- `filtered_camels_10day_attributes_cleaned_update.csv`

These files were renamed only to avoid filename conflicts during packaging.

## Environment

The scripts were prepared for Python 3 and mainly rely on the following packages:

- `pandas`
- `numpy`
- `matplotlib`
- `geopandas`
- `cartopy`
- `scipy`
- `seaborn`
- `pyproj`
- `shapely`

Depending on the script, additional standard-library modules such as `os`, `ast`, `math`, and `pathlib` are also used.

## How to Run

Run scripts from inside the `code/` folder so that relative paths resolve correctly.

Example:

```bash
cd code
python "2-Geographical location.py"
```

For scripts with command-line arguments, for example:

```bash
cd code
python "7-point_density_heatmap.py" --csv "../data/csv/attr_density_with_target.csv"
```

## Important Notes

- All input file paths in the packaged scripts are relative paths.
- The package was reorganized for portability and archiving.
- Only the CSV files and shapefile required for reading were retained in `data/`.
- Output file names in several scripts were also converted to relative paths for easier reuse on another machine.

## Suggested Citation

If you upload this package to Zenodo, you can cite the final Zenodo record DOI in your manuscript or project documentation.

