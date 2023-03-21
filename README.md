# Housing_Price_prediction
## Price prediction of a house in bengaluru city

#### Problem Statement: 
The objective of this project is to develop a predictive model to estimate the price of a residential property in the future. The price of the property is influenced by various factors such as location, number of bedrooms, number of bathrooms, and the square footage of the property.

To accomplish this, we will be using Python and its associated libraries for data analysis. The project will involve collecting and cleaning relevant data, performing exploratory data analysis, feature engineering to extract useful information, and developing a regression model using appropriate techniques such as linear regression.

Additionally, we will evaluate the performance of the model using appropriate metrics and explore various ways to improve its accuracy. The end goal is to develop a reliable model that can assist in making informed decisions related to residential property investments.

#### Data collection and preprocessing:
For data collection and preprocessing, the data was obtained from Kaggle website. The original dataset contained several feature columns such as area_type, availability, location, size, society, total_sqft, bath, balcony, and price. However, certain irrelevant features like area_type, society, balcony, and availability were removed. Additionally, any missing values in the dataset were removed.

The total_sqft column originally contained range values, which were converted into float values by taking the average of the two range values. Moreover, a new column, price_per_sqft, was created to store the price per square foot information.

To ensure data quality, outliers were detected and filtered from the dataset using appropriate techniques. The end goal of this data preprocessing step was to obtain a clean and reliable dataset that can be used for further analysis and modeling.

#### Feature Engineering:
In the feature engineering step, the pandas get_dummies method was used to create a new columns for location names.
 
Furthermore, two variables were created, namely x and y, which represent the independent and dependent variables respectively. The independent variables are the features that are used to predict the dependent variable, which is the property price.

To prepare the data for modeling, the dataset was split into training and testing sets using the train_test_split method from the sklearn library. This is a common technique used to evaluate the performance of a model on unseen data. By splitting the data into training and testing sets, we can train the model on a subset of the data and then test its performance on the remaining data.

#### Model Selection: 
In model selection, the goal is to identify the best model that fits the given data. In this case, linear regression was selected as the model. To evaluate the performance of the linear regression model, cross-validation was performed using the ShuffleSplit class in scikit-learn.

#### Results and Insights:
Based on these results, it can be inferred that the property prices in the given dataset are largely influenced by factors such as location, number of bedrooms, baths, and square footage. The linear regression model can be used to predict the prices of new properties based on these factors.

Furthermore, the results of the analysis can be used to gain insights into the real estate market. For example, by analyzing the coefficients of the linear regression model, it may be possible to identify which factors have the greatest impact on property prices. This information can be used by real estate agents, investors, and other stakeholders to make informed decisions.

#### Libraries used:
* Import pandas as pd
* Import matplotlib.pyplot  as plt
* Import numpy as np
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LinearRegression
* from sklearn.model_selection import ShuffleSplit
* from sklearn.model_selection import cross_val_score

