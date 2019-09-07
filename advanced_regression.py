#DESIGN RECIPE
#this is a collaborative effort
#In this python script, we build stacked models to be trained to predict
#dependent variables with numerous advanced independent variables.

#import important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew


#read in csv file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#viewing information about the dataset to determine next steps needed
print(train.shape)
print(test.shape)

#save the ID column to be reassigned to the output file in the end
train_ID = train["Id"]
test_ID = test["Id"]

#removing the ID column from the dataset as it is not an independent variable
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#building graphs and other visuals to further examine how the independent
#variables are affecting the dependent variable. Check for outliers!
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'],train['SalePrice'])
plt.ylabel("SalePrice", fontsize=21)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#QUICK NOTE: From examining the plot we do see some outliers.
#We will remove some outliers that are skewing the result later.


#building visuals and graphs to examine the dependent variable. determine if
#it is necessary to apply any transformations.
sns.distplot(train['SalePrice'], fit=norm);

#we examine that it is left skewed

#output mean and standard deviation
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#build the legend
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#for an additional check, get the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show()


#from the above information obtained, we realize we should apply a log transformation to normalize the dataset

train['SalePrice'] = np.log1p(train["SalePrice"])

#display the distribution:
sns.distplot(train['SalePrice'], fit = norm);

#get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel = ('Frequency')
plt.title('SalePrice distribution')
#get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#with this being said, we see a much better fit.


#this is when we start to feature engineer:
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)
print('all_data size: {}'.format(all_data.shape))

#checking for missing data:
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

#we see that poolQC, MiscFeature, Alley, Fence, FireplaceQu, and lot Frantage
# are the ones with a lot of missing values.
#please make a decision here on whether or not to replace it.

'''
#to make this decision, we can graph it!
f, ax = plt.subplots(figsize = (15, 12))
plt.xticks(rotation='90')
sns.barplot(x = all_data_na.index, y = all_data_na)
plt.xlabel('Features', fontsize = 31)
plt.ylabel('Percent of missing values', fontsize = 31)
plt.title('percent missing data by feature', fontsize = 31)
'''


#use a correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax = 0.9, square = True)


#time to fill in NA values:
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna("None")
all_data["FireplaceQu"] = all_data['FireplaceQu'].fillna('None')


#At this stage, we are to fill in missing data

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

#fill in more
for col in('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

#since we see that no garage means no cars in such garage:
for col in('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#when there is no basement these are all 0s
for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[i] = all_data[i].fillna(0)

#when there is no basement all these are 0s as well:
for col in('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

#fill in the Type to be "None" and the area to be 0s as well:
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#this is where we could be insanely wrong, but we are going with it.
#since RL is the most common value, we are going to fill in missing values with RL

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#dropping useless columns
all_data = all_data.drop(['Utilities'], axis = 1)

#more following data descriptions
all_data["Functional"] = all_data["Functional"].fillna('Typ')

# we found 1 na value in electrical column, just fill it with the most likely outcome
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# similarly for kitchen qual we only got one unfilled, its probably the most common one anyway. so we filled it.
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None')

# that took a lot of manual work. it was tiring but it was necessary. This is when we complete the whole
# process with a quick check on if there is any other leftover values.

all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())

#from this output, we can see that there is no missing value. we have finally cleansed it!

#this is where we can do more feature engineering.
#MSSubClass = The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable:
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features:
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


#label encoding some categorical variables that may contain information in their ordering set:
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

#process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

#shape:
print('Shape all_data: {}'.format(all_data.shape))

#mechanical thinking here: we all total area of basement:
#adding total sqfootage feature:
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#skewed features:
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index

#cheak the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

#To deal with highly skewed features, we compute a Box Cox Transformation (need to learn this)
skewness = skewness[abs(skewness)>0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

#all_data[skewed_features] = np.log1p(all_data[skewed_features])

#here we get a dummy categorical feature:
all_data = pd.get_dummies(all_data)
print(all_data.shape)
print(all_data.columns.values)


#splitting the all_data variable back to separate datasets we needed to train and test the model.
train_size = train.shape[0]
train = all_data[:train_size]
test = all_data[train_size:]


#HERE COMES OUR MODELLING PART!

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#validation function:
#we need this cross validation function to check for suitability in prediction models.
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state = 42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv = kf))
    return(rmse)

#base model:
#LASSO regression:
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

#Elastic net regression:
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005, l1_ratio = 0.9, random_state=3))

#Kernel ridge regression:
KRR = KernelRidge(alpha = 0.6, kernel = 'polynomial', degree = 2, coef0 = 2.5)

#Gradient boosting regression:
GBoost = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05,
                                   max_depth=4,
                                   max_features = 'sqrt',
                                   min_samples_leaf = 15,
                                   min_samples_split=10,
                                   loss = 'huber',
                                   random_state = 5)

#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma = 0.0468,
                             learning_rate = 0.05,
                             max_depth=3,
                             min_child_weight = 1.7817,
                             n_estimators = 2200,
                             reg_alpha = 0.4640,
                             reg_lambda= 0.8571,
                             subsample = 0.5213,
                             silent = 1,
                             random_state = 7,
                             nthread = -1)

#LightGBM
model_lgb = lgb.LGBMRegressor(objective = 'regression', num_leaves = 5,
                              learning_rate = 0.05,
                              n_estimators=720,
                              max_bin = 55,
                              bagging_fraction=0.8,
                              bagging_freq = 5,
                              feature_fraction = 0.2319,
                              feature_fraction_seed = 9,
                              bagging_seed = 9,
                              min_data_in_leaf=6,
                              min_sum_hessian_in_leaf = 11)

#Base models scores:
score = rmsle_cv(lasso)
print('\nLasso score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#stacking models:
#basics: just average the base models.

#build a new class to extend scikit-learn:

class AveragingModels (BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    #we define clones of the original models to fit the data in.
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        #we then train the cloned base models
        for model in self.models_:
            model.fit(X,y)

        return self

    #we then do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
                model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


#Here we can just average base models score.
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(
        score.mean(), score.std()))

#--------------------------------------------/
#THIS IS THE MAIN MODEL USED. THE MORE COMPLEX MODEL
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    #like before, we fit the data on clones of the original models:
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle = True, random_state=156)
        #Train cloned base models then create out-of-fold predictions
        #that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
            #now train the cloned metal-model using the out-of-fold predictions
            #as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
        #do the predictions of all base models on the test data and use the
        #averaged predictions as meta-features for the final prediction which
        #is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models
                                        ]).mean(axis = 1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


#NOW LETS CHECK THE SCORE OF THE COMPLEX MODEL:
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(),score.std()))

#we define the evaluation function:
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#final training and predictions:
#stackedRegressor:
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

#XGBoost:
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15))

#FINALLY
#ENSEMBLE PREDICTION:
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

#write to file
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('final_answers.csv', index = False)
