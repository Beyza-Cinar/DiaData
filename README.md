# DiaData: An Integrated Large Dataset for Type 1 Diabetes and Hypoglycemia Research

This project presents DiaData, an integrated dataset that combines Continuous Glucose Monitoring (CGM) measurements from 15 different datasets, all collected from patients with Type 1 Diabetes. The datasets used in this study were obtained from a variety of third-party sources. Due to licensing restrictions, we are unable to redistribute the full integrated dataset. However, this repository provides:

- A detailed description of the code and functions used to retrieve and process data from each individual dataset.
- Access to a partial version of the integrated dataset, containing 13 out of the 15 datasets. The data can be found in https://www.kaggle.com/datasets/beyzacinar22/diadata.

## Requirements

python version 3.11.4

matplotlib version 3.7.2

pandas version 2.0.3

numpy version 1.24.3

polars version 1.27.1


## Summary of Used Datasets 

| Dataset                                                      | CGM Frequency | Subject Counts | Availability |
| ------------------------------------------------------------ | ------------- | -------------- | ------------ |
| T1DiabetesGranada                                            | 15            | 736            | Restricted   |
| RT-CGM                                                       | 5/10          | 451            | Free         |
| ReplaceBG                                                    | 5             | 226            | Free         |
| WISDM                                                        | 5             | 206            | Free         |
| SHD (Severe Hypoglcemia Dataset)                             | 5             | 200            | Free         |
| DLCP3                                                        | 5             | 169            | Free         |
| CITY                                                         | 5             | 153            | Free         |
| SENCE                                                        | 5             | 144            | Free         |
| PEDAP                                                        | 5             | 103            | Free         |
| DiaTrend                                                     | 5             | 54             | Restricted   |
| HUPA-UCM                                                     | 15            | 25             | Free         |
| DDATSHR (dataset diabetes adolescents time-series with heart-rate) | 5/15    | 18             | Free         |
| ShanghaiT1D                                                  | 15            | 16             | Free         |
| D1NAMO                                                       | 5             | 9              | Free         |
| T1GDUJA                                                      | 5             | 1              | Free         |



## Sources of the Datasets

1. **T1DiabetesGranada**: Rodriguez-Leon, C., Aviles-Perez, M.D., Banos, O. *et al.* T1DiabetesGranada: a longitudinal multi-modal dataset of type 1 diabetes mellitus. *Sci Data* **10**, 916 (2023). https://doi.org/10.1038/s41597-023-02737-4.

  Steps necessary to access the T1DiabetesGranada dataset ( https://doi.org/10.5281/zenodo.10050944):

  - Register for a Zenodo account (https://zenodo.org).
  - Accept the Data Usage Agreement.
  - send a request specifying full name, email, and the justification of the data use.

2. **DiaTrend**: Prioleau T., Bartolome A., Comi R., Stanger C. "DiaTrend: A dataset from advanced diabetes technology to enable development of novel analytic solutions," Scientific Data, 10 (556), 2023. https://doi.org/10.1038/s41597-023-02469-5.

  Steps necessary to access the DiaTrend dataset on (https://doi.org/10.7303/syn38187184):

  - Register for a Synapse account (www.synapse.org).
  - Become a Synapse Certified User with a validated user profile.
  - Submit an Intended Data Use statement.
  - Agree to the Conditions of Use.

3. **CITY**: The source of a subset of the data is the Joslin Diabetes Center and/or Jaeb Center for Health Research. (2018). CGM Intervention in Teens and Young Adults with T1D (CITY) (3.0). Retrieved from https://public.jaeb.org/dataset/565. The analyses content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the Joslin Diabetes Center and/or Jaeb Center for Health Research.

4. **ReplaceBG**: The source of a subset of the data is the T1D Exchange (2015), A Randomized Trial Comparing Continuous Glucose Monitoring With and Without Routine Blood Glucose Monitoring in Adults with Type 1 Diabetes (2.0). Retrieved from https://public.jaeb.org/dataset/546. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the T1D Exchange. 

5. **RT-CGM**: The source of a subset of the data is the Jaeb Center for Health Research (2006), Randomized Study of Real-Time Continuous Glucose Monitors (RT-CGM) in the Management of Type 1 Diabetes (N/A). Retrieved from https://public.jaeb.org/dataset/563. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the Jaeb Center for Health Research. 

6. **DLCP3**: The source of a subset of the data is the University of Virginia and Jaeb Center for Health Research (2018), The International Diabetes Closed Loop (iDCL) trial: Clinical Acceptance of the Artificial Pancreas A Pivotal Study of t:slim X2 with Control-IQ Technology (10.0). Retrieved from https://public.jaeb.org/dataset/573. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the University of Virginia and Jaeb Center for Health Research. 

7. **SENCE**: The source of a subset of the data is the Jaeb Center for Health Research (2017), Strategies to Enhance New CGM use in Early childhood (SENCE): A Randomized Clinical Trial to Assess the Efficacy and Safety of Continuous Glucose Monitoring in Youth < 8 with Type 1 Diabetes (3.0). Retrieved from https://public.jaeb.org/dataset/537. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the Jaeb Center for Health Research

8. **SHD**: The source of a subset of the data is the T1D Exchange (2013), Severe Hypoglycemia in older adults with Type 1 Diabetes (2.0). Retrieved from https://public.jaeb.org/dataset/537. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the T1D Exchange.  

9. **WISDM**: The source of a subset of the data is the T1D Exchange (2018), Wireless Innovation for Seniors with Diabetes Mellitus (WISDM) (3.2). Retrieved from https://public.jaeb.org/dataset/564. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the T1D Exchange. 

10. **PEDAP**: The source of a subset of the data is the University of Virginia, Barbara Davis Center, University of Colorado And Jaeb Center for Health Research (2021), The Pediatric Artificial Pancreas (PEDAP) trial: A Randomized Controlled Comparison of the Control- IQ technology Versus Standard of Care in Young Children in Type 1 Diabetes (11.0). Retrieved from https://public.jaeb.org/dataset/599. The analyses, content, and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by University of Virginia, Barbara Davis Center, University of Colorado And Jaeb Center for Health Research

11. **D1NAMO**: Fabien Dubosson, Jean-Eudes Ranvier, Stefano Bromuri, Jean-Paul Calbimonte, Juan Ruiz, & Michael Schumacher. (2018). The open D1NAMO dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management (1.2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5651217

12. **HUPA-UCM**: Hidalgo, J. Ignacio; Alvarado, Jorge; Botella, Marta; Aramendi, Aranzazu; Velasco, J. Manuel; Garnica, Oscar (2024), “HUPA-UCM Diabetes Dataset”, Mendeley Data, V1, doi: 10.17632/3hbcscwz44.1

13. **DDATSwHR**: https://github.com/ictinnovaties-zorg/dataset-diabetes-adolescents-time-series-with-heart-rate/tree/main/data-csv
    
15. **ShanghaiT1D**: Zhu, Jinhao (2022). Diabetes Datasets-ShanghaiT1DM and ShanghaiT2DM. figshare. Dataset. https://doi.org/10.6084/m9.figshare.20444397.v3
    
17. **T1GDUJA**: Gaitán Guerrero, J. F., López Ruiz, J. L., Martínez Cruz, C., & Espinilla Estévez, M. (2024). T1GDUJA: Glucose dataset of a patient with type 1 diabetes mellitus [Data set]. In IEEE Internet of Things Journal. Zenodo. https://doi.org/10.5281/zenodo.11284018



## Code Organization

The code is organized as follows: 

- datasets for T1D.zip: Contains the integrated dataset. If restricted datasets are downloaded, they should be stored in this folder under datasets for T1D/granada or datasets for T1D/DiaTrend. The integrated dataset can be downloaded from https://www.kaggle.com/datasets/beyzacinar22/diadata,
- data_intergation.py: Contains the functions to read and integrate the single datasets. If necessary, the paths to the datasets should be changed here. 
- Data_Exploration.ipynb: Contains example code to read the required funtions. Provides statistical analysis of the dataset.
