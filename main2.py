# Import function to create training and test set splits
from sklearn.model_selection import train_test_split
# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np


df = pd.read_csv('data2.csv')


# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.1
lasso_nalpha=1
lasso_iter=100000
# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 8
# Test/train split
train, test = train_test_split(df, test_size = 0.20, random_state = 100) 
print (train.to_numpy())
print('------------------')
print (test.to_numpy)

#X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'],test_size=test_set_fraction)
# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for degree in range(degree_min,degree_min+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
    #print(train[['date','lat','lon','zone']].to_numpy())
    #print('-----------------------')
    #print(train['deaths'].to_numpy())
    #print('-----------------------')
    model.fit(train[['date','lat','lon','zone']].to_numpy(),train['deaths'].to_numpy())
    i=0
    for index,row in test.iterrows():
        if (i>10):
            break
        print(row)
        temp = row[['date','lat','lon','zone']].to_numpy()
        temp = temp.reshape(1,-1)
        print(model.predict(temp))
        print(row['deaths'])
        i+=1
    #print(model.predict(test[['date','lat','lon','zone']].to_numpy()))
    #tester_data = test.head(1)[['date','lat','lon','zone']].to_numpy()
    #tester_output = test.head(1)['deaths']
    #test_pred = np.array(model.predict(tester_data))
    #print(tester_data)
    #print(test_pred)
    #print(tester_output)
    
    #print(test)
    #test_pred = np.array(model.predict(test[['date','lat','lon','zone']].to_numpy()))
    
    #print('##########################################3')
    
    #RMSE=np.sqrt(np.sum(np.square(test_pred-test['deaths'].to_numpy())))
    #print(test_pred)
    test_score = model.score(test[['date','lat','lon','zone']].to_numpy(),test[['deaths']].to_numpy())
    #print(test_score)