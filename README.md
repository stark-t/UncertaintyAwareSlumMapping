# UncertaintyAwareSlumMapping

Work in progress repository!

Welcome to the official repository for the paper "Uncertainty Aware Slum Mapping in 55 Heterogeneous Cities".

## Abstract

Slums are densely populated urban areas characterized by substandard housing and squalor. These areas often lack basic infrastructure and services, making them challenging to manage and improve. Mapping slums on a large scale is particularly difficult due to their complex, dynamic, and non-uniform nature and the scarcity of available data. This research leverages advanced machine learning techniques and uncertainty aware methodologies to map slum areas across 55 heterogeneous cities. We effectively address the challenges posed by limited labeled data and achieve robust slum probability maps. Our coherent methodology, applied to a large slum dataset across the Global South, includes probability estimates for each prediction, offering a detailed understanding of slums within each city. These insights offer a spatially detailed map of slums and the multiple facets that come with slum settlements and their probabilities. By providing a nuanced view of slum distributions, our work highlights the diversity and complexity of slum settlements, contributing to a more comprehensive understanding of these areas. A significant achievement of our research is detecting various slum categories, depending on their set of slum morphology features, and their different probabilities, especially in cases where slum settlements gradually transition into formal settlements or display atypical characteristics. This approach offers a substantial improvement over traditional binary slum classification methods that focus solely on typical slum morphologies.

## Project Guide

### **Install Repository Dependencies**

1. install requirements.txt
2. change config examples and adjust paths accordingly

### **Get Data**

**Note:** The data provided here is solely intended for illustrative purposes and deviates from the original paper. In the original study, we utilized 4.77-meter PlanetScope data. However, due to copyright restrictions, we have substituted it with resampled RGB Sentinel-2 imagery from the cities of Caracas and Mumbai as a demonstrative example.

To use the example data, follow these steps:

1. Download the data from Figshare https://figshare.com/articles/dataset/Dataset/24988959.
2. Extract the data into the `/data/` directory.

To employ your custom dataset, it is crucial to adhere to a specific data structure. For each area of interest (AOI) within the data directory, three requisite files are essential, each sharing identical resolution and extent:

1. **Remote Sensing Imagery:** This should be in RGB format and resampled to a 4.77-meter resolution. It must be named planet_AOI.tif (e.g., planet_mumbai.tif).

2. **LCZ reference Mask:** In our case, we employed Local Climate Zones as delineated by Zhu et al., 2019. The data must be named lcz_AOI.tif (e.g. lcz_mumbai.tif).

3. **MUA Mask:** In our case, we employed morphological urban areas as delineated by Taubenböck et al., 2019. The data must be named mua_AOI.tif (e.g. mua_mumbai.tif).

4. **Slum Reference Mask:** Employ values of 1 to represent slum areas. The data must be named reference_AOI.tif (e.g. reference_mumbai.tif).

Ensuring uniform resolution and extent across all files is imperative for seamless integration into our processing pipeline.

<small><i>Zhu, X. X., Hu, J., Qiu, C., Shi, Y., Kang, J., Mou, L., ... & Wang, Y. (2019). So2Sat LCZ42: A benchmark dataset for global local climate zones classification. arXiv preprint arXiv:1912.12171.</i></small>

<small><i>Taubenböck, H., Weigand, M., Esch, T., Staab, J., Wurm, M., Mast, J., Dech,
S., 2019. A new ranking of the world’s largest cities—do administrative
units obscure morphological realities? Remote Sens. Environ. 232
111353. doi:https://doi.org/10.1016/j.rse.2019.111353.<small><i>


### **Run Code**

1. **run_split_raster.py**
    - This script splits the remote sensing data into small tiles and creates the labels.
    - The labels are created for 5 classes: 0 background, 1 urban built-up areas, 2 vegetation, 3 water and class 4 for slums.
    - All data should be of the same extent and resolution.
    - Labels are added to the image tile file name and saved in the `data/pretrain` or `data/transfer` directory.

2. **run_train.py**
    - This script pretrains the model using the example data.
    - The pretraining is performed on the Caracas dataset.

3. **run_train.py**
    - This script transfer learns the model using the example data.
    - The pretrained model is transfer-learned on the Mumbai dataset.

4. **run_predict.py**
    - This script creates results and maps using the Mumbai dataset.
    - Note: Since the example dataset is small, the same data is used for both transfer-learning and testing.

## Results

### Example results

**Note:** Please note that the results shown here are based on a limited amount of data and may not be ideal. They should be used as a reference for structuring other data to achieve similar results as described in the original paper. If you require additional data or access to the original model weights, feel free to contact the authors. They will be happy to assist you.