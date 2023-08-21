## Machine Learning model for  Fairfax County, VA, Home Sale Prices
![image](https://github.com/rvafaeis/project_4_ML_Fairfax_home_price/assets/118146659/80710ade-c8ed-49f9-9722-b8d55681920b)

The aim of this project was to gather and analyze data from local single family home sold prices in Fairfax County, VA and use that data to construct machine learning model to accurately predict home prices in fairfax county.

### Exploratory Data Analysis (EDA)
First, the data was obtained from the MLS website. The data from 2019-2023 was downloaded as .csv files and used the Python library, Pandas to merge (concatenate), explore, and clean the data. The final cleaned data was ploted using seaborn library heatmap function:

![image](https://github.com/rvafaeis/project_4_ML_Fairfax_home_price/assets/120426753/999610c7-53b9-4306-8f71-dffce88c4c02)


The interquartile range (IQR) was calculated from the statistics summary (using the .describe()) of the cleaned data to determine the potential outliers. Based on the IQR, the necessary rows of data were removed so that there won't be data imbalance.  Utilizing the IQR to remove outliers greatly improved our Neural Network model performance. These box plots show sold price outliers (left plot) and after removing the outliers (right plot).

![image](https://github.com/rvafaeis/project_4_ML_Fairfax_home_price/assets/120426753/05a5fe0b-9893-493c-b741-49a2bd57ecf8) 

A geoapify API was utilized to obtain (convert) the street addresses of the houses to its corresponding Latitude and Longitude locations.  Lastly, the get_dummies function from Pandas was implemented to convert categorical records ( in this case the unique City names) to labels for the ML algorithm.  indicate our categorical data using numbers to prepare our dataset for our machine learning model


### Neural Network
The aim of our neural network is to create a ML model that can  predict house prices in Fairfax County. To do that, a deep learning linear regression neural network model with three hidden layers was constructed. Identifying the target variable as the home sold price and the rest of the variables as features. A StandardScaler was used to reduce the overall likelihood that outliers, variables of different units, or skewed distributions will have a negative impact on a model’s performance.
The data was split into test and train variables using sklearn to begin our prediction model. 

Multiple models, activation functions, and optimizers were tested and after trial and error, the sequential model with an Early_Stopping method was the best performing model with highest r-squared value. The activation function was determined to be best as rectified linear unit (Relu) for each of the three layers in our model. Our output layer used a linear activation function and the best optimizer was determined to be rmsprop. The table shows how each of the models performed using accuracy metrics including r-squared values.
![image](https://github.com/rvafaeis/project_4_ML_Fairfax_home_price/assets/120426753/37f7888b-f9f2-4c34-a637-c790b5582565)


https://public.tableau.com/app/profile/raheleh.vafaei.saadi/viz/FairfaxVASFHomePrice/Sheet7?publish=yes
