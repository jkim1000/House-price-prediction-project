# Iowa Housing Price Prediction

**Introduction**

For this project, the primary objective was to create and assess regression models to accurately predict house prices based on a Kaggle competition dataset, which is available here: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview). The dataset includes around 3000 records of house sales in Ames, Iowa between 2006 – 2010 and contains 79 explanatory variables detailing various aspects of residential homes such as square footage, number of rooms and sale year. The data is split equally into a training set, which will be used to create the model and a test set, which will be used to test model performance. The general workflow to create the model will be as follows:

1.	Data preprocessing
2.	Exploratory data analysis/Feature Engineering
3.	Model training & hyperparameter tuning
4.	Model diagnostics & evaluation
5.	Result interpretation

 ![image](https://user-images.githubusercontent.com/67875208/118532144-9e6b6780-b714-11eb-9b4f-3a725e279a75.png)

 
**Data preprocessing**

The first step before implementing machine learning models was to preprocess the data by analyzing the different types of features in the dataset, imputing missing values and removing outliers. These preprocessing steps are integral to model performance later on as they improve the quality and interpretability of the dataset. 
Out of the 80 different explanatory variables, 23 were nominal, 23 were ordinal, 14 were discrete and 20 were continuous variables. Among those, features with more than 80% missing observations such as ‘PoolQC’, ‘MiscFeature’, ‘Alley’ and ‘Fence’ were dropped completely. After careful reading of the documentation on each of these variables, many ‘NA’ values corresponded to the absence of a feature and not necessarily a missing value. Therefore, I replaced these ‘NA’ values with either ‘None’ for categorical variables or ‘0’ for numerical variables. For the ‘Lot Frontage’ feature, a KNN method was used to impute the missing values instead of the mean/mode because mean/mode imputations ignore feature correlations and reduces variance of the data. 
Once missing values were imputed, categorical features with string values were converted into ordinal variables by mapping them to integers. For instance, evaluative features containing rankings from ‘Poor’ to ‘Excellent’ were mapped to 1-5 to enhance the data interpretability (i.e. {‘None’:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}. Furthermore, categorical features containing ordinal data (i.e. ‘LotShape’, ‘BldgType’, ‘BsmtExposure’) were converted into ordinal features accordingly by mapping them into a numerical format. Some numerical variables such as ‘MoSold’ and ‘YrSold’ were converted to string type since they do not truly reflect numerical properties. 

**Exploratory data analysis**

Based on the density plot of the dependent (target) variable, Sale Price, it was immediately apparent that it was right skewed. Since multivariate normality is one of the assumptions for linear regression, a log transformation was applied to fix the skewness, which yielded a much more normal distribution (as shown below). 

![image](https://user-images.githubusercontent.com/67875208/118532180-a925fc80-b714-11eb-9c34-aaf7bf95f4d2.png)

![image](https://user-images.githubusercontent.com/67875208/118532195-acb98380-b714-11eb-8903-4d5965dd1894.png)


Once the Sale Price was transformed, a correlation matrix was used to try to identify multicollinearity among the variables (Figure 3). In order to reduce multicollinearity among the explanatory variables, several new features were created that encapsulated information of several others such as:
‘Total Bath’ – total number of bathrooms in the house/basement
‘Overall Score’ – total score combining overall quality and overall condition
‘House age’ – number of years from remodeling to sale
‘Overall Porch’ – total square feet area of porch
‘Basement Area’ – total square feet area of basement
New features that indicate the presence of house features such as a pool or a fireplace were created and imputed into binary values of either 0 (doesn’t exist) or 1 (exist). 

Applying a correlation heatmap to the resultant engineered data reveals which features are most correlated to the sale price (SalePrice) and which are least correlated according to Spearman’s correlation coefficients. Features such as Overall Quality and Above Ground Living Area were highly correlated with Sale Price, in contrast with the Month or Year Sold, which showed near zero correlation with Sale Price. Among the highly correlated variables, outliers that were either above or under than 3 standard deviations from the mean were removed from the dataset since outliers increase the error variance and could potentially skew our model predictions.

![image](https://user-images.githubusercontent.com/67875208/118532222-b5aa5500-b714-11eb-9cf8-835f19dc6c17.png)

 
For certain explanatory variables in the dataset, features that displayed more than 90% dominance in one category (i.e. 'Street', 'LandContour', 'Utilities') were dropped completely as they would not provide any predictive/explanatory value in creating the model. For the ‘Neighborhood’ feature, the different areas were subdivided into ‘high’, ‘middle’ and ‘low’ priced neighborhoods to prevent too much expansion of feature space after dummification. Once the categorical variables were processed appropriately, they were dummified in order to be able to implement the linear regression models, meaning that meaning they were converted into a series of binary variables representing whether each category was true for a given house, yes (1) or no (0).

**Model training **

We explored two types of models in this study: linear regression models and tree-based ensemble models. 
For linear regression, we took the log value of the target variable, Sale Price and trained the model with multiple linear regression and regularization models such as Ridge and Lasso using a 70-30 train validations split. All three linear models provided train-test scores of 0.90–0.91, MSE of approximately 0.013, and RMSE of approximately 0.114. Then the model was validated using 5fold cross validation, yielding a mean cross validation score of approximately 0.88. Results of the models are summarized below:

![image](https://user-images.githubusercontent.com/67875208/118532357-df637c00-b714-11eb-9773-6a2d716c1682.png)


As we can see, all three linear models produced relatively consistent results, with Ridge regression performing slightly better than the rest. The model coefficients suggest that the overall score (positive), fireplace (positive), kitchen/garage quality (positive), neighborhood (negative) and house age (negative) are among the most significant variables affecting the overall sale price of the house. 

![image](https://user-images.githubusercontent.com/67875208/118532379-e5595d00-b714-11eb-814a-1259999e1f75.png)

 
Normal distribution and independence of residuals or error terms from predictors are core assumptions of all regression models. The skewness and kurtosis analyses confirm that these assumptions hold true as they indicate that the error distributions are approximately zero with a mean of zero. The residual plot further confirms these assumptions as we see a nice even distribution around zero. 

![image](https://user-images.githubusercontent.com/67875208/118532392-ea1e1100-b714-11eb-854b-c57042c3acfe.png)


The quantile-quantile (qq) plot is a graphical technique for determining if two data sets come from populations with a common distribution. Here, a qq plot allows us to determine whether the data is normally distributed, and in this case, our plot is relatively straight between the 25% - 75% quartile range. However, the qq plot indicates that the ridge model tends to underpredict house prices in the higher quantiles and overpredict those in the lower quantiles. Therefore, we conclude that the model is not as robust in predicting house prices when dealing with data points at the extremes of the price range. 

![image](https://user-images.githubusercontent.com/67875208/118532450-fe620e00-b714-11eb-9786-2a021ed408aa.png)


In order to quantify the effect of each feature in our regression model, we computed the exponent of each coefficient (since the target variable is in natural log form) in order to calculate the dollar change in the average house sale price given a unit change of a given explanatory variable. Results are shown in the table below:

![image](https://user-images.githubusercontent.com/67875208/118532491-0621b280-b715-11eb-90f6-3d160622386e.png)

![image](https://user-images.githubusercontent.com/67875208/118532519-0c179380-b715-11eb-890e-4265956127be.png)


For example, we can see that for a given unit change of HouseAge, the sale price decreases by $1160 while for a given unit change of the OverallScore or Fireplace, the sale price increases by $6617 and $5155 respectively. 

However, one drawback of the regression model is that it contains dummified features that obscure the interpretability of the explanatory variables. Therefore, in addition to the linear models implemented above, we also tried non-linear, tree-based, models. Furthermore, there may be non-linear relationships in our data that tree-models are able to capture, perhaps offering better performance than the linear models described above. RMSE scores for the non-linear models are summarized below:

![image](https://user-images.githubusercontent.com/67875208/118532556-146fce80-b715-11eb-84ce-4eecd44e3abc.png) 


![image](https://user-images.githubusercontent.com/67875208/118532623-2c475280-b715-11eb-8799-3e604bfceb9a.png)


![image](https://user-images.githubusercontent.com/67875208/118532646-3406f700-b715-11eb-88bd-31acb57183be.png)


![image](https://user-images.githubusercontent.com/67875208/118532672-39644180-b715-11eb-8501-0b0b3b656564.png)


![image](https://user-images.githubusercontent.com/67875208/118532705-41bc7c80-b715-11eb-890d-5a5a067743cd.png)

 
**Conclusion**

These results can help inform decision-making at the business level. As stated above, it can provide insight on the pricing of real estate assets just by plugging in the house characteristics and letting the model return a price. In addition, it can provide information on which features of a new house are more valuable for potential house buyers.
Among the models that we have explored in this project, we find that ridge regression performs best for linear regression models while XGBoost works best for nonlinear models based on the RMSE and R^2 scores. Based on these model outputs, we conclude that fireplace, total square footage, size of the garage (GarageCars) are among the most significant positively correlated features that affect the sale price. 

Choosing the optimal model ultimately depends on what we are trying to achieve through machine learning. For instance, if the primary goal is to predict the sale price of a house, a simple linear model in this case would suffice. To improve the predictive accuracy of the outliers observed in our data, one might consider using a stacked model combining ridge regression and XGBoost at the cost of model interpretability. However, predicting the sale price of a house isn't necessarily everything. In fact, knowing what factors influence the selling price of a house may well be more valuable than the predictions themselves from the seller’s standpoint. Therefore, a XGBoost model or even a random forest model (with better tuning) may be better options if the goal is to enhance model interpretability. 

The following are some of the insights gleaned from the outputted feature importances of each model as well as some recommendations for residential real estate buyers and sellers:

**XGBoost: Fireplace, GarageCars, TotalSF are most important**

•	Sellers: Consider investing more into installation of a fireplace to fetch higher prices
•	Buyers: Consider buying houses with no/low-quality basements and excavating/enlarging/improving these yourself if cheaper in the aggregate

**Random Forests: OverallQual, TotalArea, Area1stArea2nd, LotFrontage are most important**

•	Sellers: Consider improving fireplace quality to fetch higher prices
•	Buyers: Consider buying houses with no garage and building one yourself if cheaper in the aggregate

**Gradient Boosting Regressor: OverallQual, LotArea, BsmtQual & YearBuilt are most important**

•	Sellers: Consider investing in Central Air conditioning to fetch higher prices
•	Buyers: Consider buying older houses and remodeling these yourself

**Linear Models: LotArea, Neighborhood variables are most important**

•	Sellers: Consider building an open porch to increase house value
•	Buyers: Consider building an open porch yourself to save money if cheaper in the aggregate


