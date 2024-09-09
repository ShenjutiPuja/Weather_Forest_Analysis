Research Methodology
The study was conducted by following this methodology.
After coming up with research questions, I collected data from several
online sources. After collecting the data, I preprocessed the data to
make it more workworthy. Preprocessing involves a few steps. Handle
missing values, handle categorized values, and split datasets into train
and test data.
For training data, I used 70% data of the dataset, and for testing, I
used 30% data of the dataset.
After data preparation, I tried to understand the data by Exploratory
data analysis. I explored the data statistically and visually.
By Selecting the desired models I walked through the next step of
model evaluation. Here I used four machine learning model for this
study. Which are gradient boosting, random forest, ridge regression,
and lasso regression.
By using this model I evaluated the model and chose the best model
for further work which is prediction. By using the best-fitted model a
user interface was developed to predict the future temperature. Where
as the input month and year is worked and the output is the future
average monthly temperature.
Data Collection
Here are some sample data, what was used to for this study. The
dataset contains five variables. Average temperature, month, year,
average rainfall and forest area.
Data was collected from numerous online sources.
Temperature data was taken from NASA's POWER Project.
Rainfall Data was collected from the Humanitarian Data Exchange
(HDX) website.
The forest area data was collected from World Bank Data on Forest
Area and FAO Bangladesh Country Paper.
586 sets of data are available in this dataset.
The timeline of the data is from January 1975 to October 2023.
Result and Analysis
Four machine learning models are used here to find out the best fitted
model for the dataset. The machine learning models are:
– Gradient Boosting
– Random Forest
– Ridge Regression
– Lasso Regression
By looking at the correlation matrix we can assume that there are
highly positive correlations between rainfall and temperature which is
0.71. That means if temperature is increase the rate of rainfall is
increases too.
Also, there are Correlation between year and forest area is -0.52. That
means there are negative moderate correlation between year and
forest. This indicates that over the years forest area is decreasing.
Here are the heatmaps of temperature by month and rainfall by
month. We can visualize by this two graphs that the rainfall is highest
in july month and the highest temperature is shown from April to
September month.
Result Analysis and Comparison
Models Accuracy
Gradient Boosting 96.01%
Random Forest 95.57%
Ridge Regression 94.07%
Lasso Regression 93.89%
Implementation
By using the highest accurate model a user interface has been made
to predict future data. From all four machine learning models, we got
the highest accuracy on the model gradient boosting, which was used
for further prediction implementation,
By using this interface ew can predict weather from the year 2024 to
2036. To learn the output we need to give year and month as input.
Here are some data that was predicted by using the developed UI.
Limitations
The dataset is small. By using more historical data, we were able to
achieve more accurate results.
The weather does not depend on a specific geographical area. Here in
this study, I used the data of Bangladesh. By using more geographical
range data, the model will be more accurate.
In current study may not cover emerging features and innovations in
Climate and forest area data.
Future Works
Suggest future research with a larger and more diverse dataset.
By developing automated forecasting, where no input is required, the
work will be enriched
