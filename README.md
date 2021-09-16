This code accompanies the paper **Crop loss identification at field parcel scale using satellite remote sensing and machine learning** (doi: 10.1101/2021.05.07.443072)
- preparation: folder contains code to extract data from Landsat 7 scenes to field parcel image patches into a hdf file
- 00_preprocess_data.ipynb: code to preprocss the Landsat 7 and reference data to crate NDVI time series (section 2.3).-
- 01_optimise_models.py: code use to find the optimal hyperparameters of different models (section 3.1).
- 02_compare_opt_models.py: Comparison of different optimised model (section 3.1).
- 03_exp_within_years.py: Within year classification experiment (section 3.2).
- 04_exp_btw_year_single.py: Between year classification experiment where model is trained on data from one year and tested on another (section 3.3)
- 05_exp_btw_year_multiple.py: Between year classification experiment where model is trained on all data except the test year (section 3.3)
- 06_results_ms_figures.py: replicate manuscript figures.
- 06_results_ms_tables.py: Code to replicate manuscript tables.
