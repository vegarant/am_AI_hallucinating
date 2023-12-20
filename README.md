# Code for the chapter "Am (A)I hallucinating?"

-----------------------

Figure 3: 
* `Demo_test_automap_stability.py`
* `Demo_test_automap_non_zero_mean_noise.py`
* `Demo_test_automap_stability_knee.py`
* `Demo_test_lasso_non_zero_mean_noise.py`

Figure 4:
* `Demo_test_automap_error_map.py`
* `Demo_test_lasso_error_map.py`
* `Demo_test_read_and_plot_error_maps.py`

Figure 6:
* `Demo_test_automap_stability.py`

Figure 7:
* `Compare_sampling_patterns_fourier.m`
* `script_fourier_train_network_many_sampling_rates.py`
* `script_fourier_test_network_many_sampling_rates.py`

Table 1: 
* `Demo_test_automap_compute_norms.py`

-----------------------

## Setup for AUTOMAP experiments
The data used in the paper can be downloaded from [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/am_ai_hallucinating.zip), and the AUTOMAP network weights can be downloaded [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/cs_poisson_for_vegard.h5) (3.4 GB). After downloading the data, modify the paths in the file `adv_tools_PNAS/automap_config.py` to link all relevant paths to the data. To run the stability test for the LASSO experiment, add the [UiO-CS/optimization](https://github.com/UiO-CS/optimization) and [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) packages to your Python path. 

All of these scripts have been executed with Tensorflow version 1.14.0.

## Setup for different sampling rates (Figure 7)

1. Download and extract the desired data from the [fastMRI challenge](https://fastmri.med.nyu.edu).
2. Modify the paths in `config.py`, to point towards the desired data. 
3. Run the script `data_management.py` to generate an iterable dataset.
4. Run the script `script_fourier_train_network_many_sampling_rates.py` to train a neural network at the desired sampling rate.
5. Run the script `script_fourier_test_network_many_sampling_rates.py` to evaluate the trained neural network on several images. The network will use the sampling rate it is trained on.

The PyTorch version used for this code is 1.11.0.

To create the compressive sensing reconstructions, we have used the MATLAB script `Compare_sampling_patterns_fourier.m`. This script requires you to train at least one of the networks for the different sampling rates, as the script `script_fourier_train_network_many_sampling_rates.py` stores the original MRI images as PNG files. These images are then subsequently read by the MATLAB script. 

To run the script, you need to add the [cilib](https://github.com/vegarant/cilib) and [splg1](https://friedlander.io/spgl1/) packages to your MATLAB path. 
