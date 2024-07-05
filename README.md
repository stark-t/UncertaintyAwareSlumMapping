# UncertaintyAwareSlumMapping

Work in progress repository!

Welcome to the official repository for the paper "Uncertainty Aware Slum Mapping in 55 Heterogeneous Cities".

## Abstract

Slums are densely populated urban areas characterized by substandard housing and squalor. These areas often lack basic infrastructure and services, making them challenging to manage and improve. Mapping slums on a large scale is particularly difficult due to their complex, dynamic, and non-uniform nature and the scarcity of available data. This research leverages advanced machine learning techniques and uncertainty aware methodologies to map slum areas across 55 heterogeneous cities. We effectively address the challenges posed by limited labeled data and achieve robust slum probability maps. Our coherent methodology, applied to a large slum dataset across the Global South, includes probability estimates for each prediction, offering a detailed understanding of slums within each city. These insights offer a spatially detailed map of slums and the multiple facets that come with slum settlements and their probabilities. By providing a nuanced view of slum distributions, our work highlights the diversity and complexity of slum settlements, contributing to a more comprehensive understanding of these areas. A significant achievement of our research is detecting various slum categories, depending on their set of slum morphology features, and their different probabilities, especially in cases where slum settlements gradually transition into formal settlements or display atypical characteristics. This approach offers a substantial improvement over traditional binary slum classification methods that focus solely on typical slum morphologies.

## Project Guide

### **Install Repository Dependencies**

1. install requirements.txt
2. change config_example.py to config.py and adjust paths accordingly

### **Get Data**

**Note:** The data provided here is solely intended for illustrative purposes and deviates from the original paper. In the original study, we utilized resampled 3-meter PlanetScope data. However, due to copyright restrictions, we have substituted it with 3-meter resampled RGB Sentinel-2 imagery from the cities of Caracas and Mumbai as a demonstrative example.

To use the example data, follow these steps:

1. Download the data from Figshare https://figshare.com/articles/dataset/Dataset/24988959.
2. Extract the data into the `/data/` directory.

To employ your custom dataset, it is crucial to adhere to a specific data structure. For each area of interest (AOI) within the data directory, three requisite files are essential, each sharing identical resolution and extent:

1. **Remote Sensing Imagery:** This should be in RGB format and resampled to a 3-meter resolution. It must be named AOI_3m.tif (e.g., Mumbai_3m.tif).

2. **Urban-Background Mask:** Utilize values of 0 for background and 1 for urban areas. In our case, we employed Local Climate Zones as delineated by Zhu et al., 2019. The data must be named AOI_urban.tif (e.g. Mumbai_urban.tif).

3. **Slum Reference Mask:** Employ values of 1 to represent slum areas. The data must be named AOI_slum_reference.tif (e.g. Mumbai_slum_reference.tif).

Ensuring uniform resolution and extent across all three files is imperative for seamless integration into our processing pipeline.

<small><i>Zhu, X. X., Hu, J., Qiu, C., Shi, Y., Kang, J., Mou, L., ... & Wang, Y. (2019). So2Sat LCZ42: A benchmark dataset for global local climate zones classification. arXiv preprint arXiv:1912.12171.</i></small>


### **Run Code**

1. **run_split_raster.py**
    - This script splits the remote sensing data into small tiles and creates the labels.
    - The labels are created for 3 classes: 0 and 1 for background and urban areas, and class 2 for slum polygons.
    - All data should be of the same extent and resolution.
    - Labels are added to the image tile file name and saved in the `data/datasets` directory, along with image statistics used for normalization.

2. **run_train_pretraining.py**
    - This script pretrains the STnet using the example data.
    - The pretraining is performed on the Caracas dataset.

3. **run_train_transferlearning.py**
    - This script finetunes the STnet using the example data.
    - The pretrained STnet is transfer-learned on the Mumbai dataset.

4. **run_inference.py**
    - This script creates results and maps using the Mumbai dataset.
    - Note: Since the example dataset is small, the same data is used for both transfer-learning and testing.
    - In the original paper, a 2-fold split is used for transfer-learning and testing, and the results are merged afterwards.

## Results

### Example results

**Note:** Please note that the results shown here are based on a limited amount of data and may not be ideal. They should be used as a reference for structuring other data to achieve similar results as described in the original paper. If you require additional data or access to the original model weights, feel free to contact the authors. They will be happy to assist you.