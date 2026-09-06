#  Predictive Modeling for Physical Activity & Volunteering

## Project Overview
This repository contains the code and analytical findings associated with a dissertation submitted to the University of Bristol for MSc Data Science. The project uses unsupervised learning techniques to investigate physical activity (PA) inequalities through the lens of intersectionality, across London's heterogeneous population. This analysis is based on Active Lives survey data from 2016-2023, on which we will track trends to identify subpopulations and geographies in need of attention. Furthermore, we will employ sensitivity analysis to identify subpopulations most responsive to targeted psycho-social intervention. Using both geographic and intersectional perspectives, our results indicate significant spatial and cluster-level PA inequalities, thus urgently requiring the attention of policymakers. Though we were unable to definitively ascertain the specific number of subpopulations within London, strong evidence exists of heterogeneous subpopulations within London, with socio-demographic and psycho-social features yielding non-trivial splits. Additionally, this work hopes to promote the advantages of intersectional perspectives for policy makers, enabling more targeted intervention.

### Core Findings: Boroughs Requiring Attention
* **Chronically Inactive Boroughs:** Barking and Dagenham, Bexley, Brent, Croydon, Enfield, Havering, Hillingdon, Hounslow and Newham.
* **Disguised Inactive Boroughs:** Hackney, Greenwich, Tower Hamlets, Haringey, Kensington and Chelsea, Lambeth, Lewisham, Southwark and Waltham Forest.

### Core Findings: Subpopulations Requiring Attention
* **Chronically Inactive Subpopulations:** Unemployed Low Social Group Adults, Long-term Sick and Disabled Adults and Disabled Old Retirees.
* **Disguised Inactive Subpopulations:** Low Social Group Mothers and Carers and Middle-Aged Working Fathers.

## Project Objectives

* **Objective 1:** Examine geographical disparities in physical activity and investigate differences at borough-level throughout London.
* **Objective 2:** Identify and characterise intersectional subpopulations using unsupervised clustering techniques, profiling these groupings.
* **Objective 3:** Develop forecasts for physical activity at both borough and subpopulation-level, while considering uncertainty and seasonality.
* **Objective 4:** Explore the sensitivity of distinct subpopulations to changes in different psychosocial determinants, directly informing the design of targeted interventions.

## Core Methodologies

* **Clustering:** We employ two separate clustering pipelines, harnessing the combination of UMAP-HDBSCAN and Latent Class Analysis (LCA). 
* **Forecasting:** Models such as Bayesian Linear Regression, Bounded Regression and Seasonal ARIMA were used to forecast PA trends. Logistic regression analysis was also used for exploratory coefficient investigation.
* **Sensitivity Analysis:** LightGBM, Random Forest and XGBoost classifiers were trained using Optuna for hyperparameter optimisation. Sensitivity analysis, by which a one standard deviation perturbation was applied to a selected psycho-social feature, was conducted using LightGBM.

## Repository Structure

### **src** 
The src folder is comprised of 4 main sections representing the 4 core areas of our analysis
* **stream_1:** Relates to borough level analysis conducted on the 32 London boroughs. Forecasts were developed using a Bayesian Ridge forecaster and seasonal forecasts used SARIMA.
* **stream_2:** Contains exploratory analysis relating to ols and logistic coefficient analysis for determinants of physical activity participation.
* **stream_3:** Encompasses code relating to clustering analysis for both LCA and HDBSCAN. Additionally it also contains a UMAP class which was used to generate the embeddings for both sports clustering and the UMAP-HDBSCAN pipeline. Cluster-wise forecasting methods are also included in this directory.
* **perturbation:** Contains files relating to the model training pipeline for perturbation (sensitivity analysis) pipeline. Additionally contains the perturbation algorithm used to test the trained models and the models subdirectory provides the code for the classifiers.

### **data**
The data directory contains the data used throughout this project. In particular, the master_data subdirectory contains the data used as an input to the clustering pipeline and the output which contains the generated cluster labels. Additionally, the test data used for the sensitivity analysis is also contained within the perturbation subdirectory.

### **results**
Contains the results for the sensitivity analysis across all of the tested models and LCA outputs. Additionally, it also contains the saved umap embeddings used for the n neighbors search, and the optimised hdbscan instance labels saved in dictionary format in hdb_cluster_dict.pkl.

### **figures**
This folder contains all figures generated throughout the project and used in the final report.

### **trained_models** 
Saved models relating to sensitivity analysis, and the trained stepmix (LCA) instance are saved in this folder.

### **notebooks**
Comprised of exploratory or test notebooks used to develop files in the src folder.

### **visualisation**
An additional Tableau visualisation had been added to aid further investigation by policymakers.

## References

* Frenzel, C. (2025) 
    Tuning with HDBSCAN, Towards Data Science. 
    Available at: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970/ (Accessed: 27 August 2026). 

* McInnes, L. (2018)
    Combining multiple UMAP models - umap 0.5.8 documentation. 
    Available at: https://umap-learn.readthedocs.io/en/latest/composing_models.html (Accessed: 27 August 2026). 

* Zouinina, S. (2024)
    A deep dive into LIGHTGBM: How to choose and tune parameters, Medium
    Available at: https://medium.com/@sarahzouinina/a-deep-dive-into-lightgbm-how-to-choose-and-tune-parameters-7c584945842e (Accessed: 27 August 2026). 