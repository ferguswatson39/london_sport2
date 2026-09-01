#  London Sport: Predictive Modeling for Physical Activity & Volunteering

data - https://datacatalogue.ukdataservice.ac.uk/series/series/2000120#access-data

A machine learning project focused on forecasting physical activity participation, volunteering trends, and activity patterns across London's boroughs over the next decade.

---

##  Project Overview
London Sport seeks to strengthen its forecasting capabilities by building a clearer understanding of future scenarios that may affect Londoners’ participation in sport, physical activity, and volunteering. 

By leveraging machine learning techniques and data from **Sport England’s Active Lives Survey**, this project explores complex interactions between demographic, environmental, and behavioral factors to anticipate operational and behavioral patterns over the next **ten years**.

---

##  Core Analysis Areas

### 1. Overall Participation Forecasts
* **Macro Trends:** Projected levels of physical activity and sport-specific club membership across London as a whole over the next ten years.
* **Activity Types:** The likelihood of Londoners participating in specific sports and activities over the next decade.
* **Geographic Mapping:** Granular forecasts of activity levels mapped individually for each London borough.
* **Time Barriers:** Identification of key factors that may reduce individuals’ available free time and subsequently affect participation rates.

### 2. Demographic-Specific Insights
* **Borough Intersectionality:** Projected likelihood of participation in specific activities within each individual London borough.
* **Targeted Cohorts:** Forecasted participation levels and specific sport preferences broken down by:
  * Age
  * Gender
  * Ethnicity
  * Disability status
  * Socio-economic groups
* **Health Metrics:** Likelihood of physical activity participation among individuals living with general or long-term health conditions.

### 3. Indoor vs. Outdoor Activity
* **Environmental Balance:** The expected split and structural balance of indoor versus outdoor physical activity over the next ten years.
* **Preference Dynamics:** Differences in indoor and outdoor activity environments among people with varying baseline activity levels (from highly active to least active).
* **Regional Disparities:** Statistical relationships between inner vs. outer London locations, baseline activity levels, and socio-economic factors.

### 4. Volunteering in Sport
* **Inclusion Metrics:** The likelihood of individuals from diverse backgrounds volunteering in sport and physical activity over the next decade.
* **Role Distribution:** Forecasted distribution of specific volunteer roles that different demographic groups (including ethnicity, disability status, gender, and age) are most likely to take on.

---

## Repository Structure

### **src** 
The src folder is comprised of 4 main sections representing the 4 core areas of our analysis
* **stream_1:** Relates to borough level analysis conducted on the 32 London boroughs. Forecasts were developed using a Bayesian Ridge forecaster and seasonal forecasts used ARIMA.
* **stream_2:** Contains exploratory analysis relating to ols and logistic coefficient analysis for determinants of physical activity participation.
* **stream_3:** Encompasses code relating to clustering analysis for both LCA and HDBSCAN. Additionally it also contains a UMAP class which was used to generate the embeddings for both sports clustering and the UMAP-HDBSCAN pipeline. Cluster-wise forecasting methods are also included in this directory.
* **perturbation:** Contains two files relating to the model training pipleine for perturbation (sensitivity analysis) pipeline. Additionally contains the perturbation algorithm used to test the trained models.

### **data**
The data directory contains the data used throughout this project. In particular, the master_data subdirectory contains the data used as an input to the clustering pipeline and the output which contains the generated cluster labels. Additionally, the test data used for the sensitivity analysis is also contained within the perturbation subdirectory.

### **results**
Contains the results for the sensitivity analysis across all of the tested models. Additionally, it also contains the saved umap embeddings used for the n neighbors search, and the optimised hdbscan instance labels saved in dictionary format in hdb_cluster_dict.pkl.

### **figures**
This folder contains all figures generated throughout the project and used in the final report.

### **models** 
Saved models relating to sensitivity analysis, and the trained stepmix (LCA) instance are saved in this folder.

### **notebooks**
Comprised of exploratory or test notebooks used to develop files in the src folder.