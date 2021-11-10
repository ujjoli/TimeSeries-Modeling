import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from pandas.plotting import register_matplotlib_converters
from numpy import linalg as LA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import chi2
from scipy import signal

register_matplotlib_converters()


Amzn_df = web.DataReader('AMZN', data_source='yahoo',start ='2000-01-01', end='2021-03-01')
len(Amzn_df)
depend_var = Amzn_df['Close']

print(len(Amzn_df))

#5.a. Plot of the dependent variable versus time.
plt.plot(depend_var, label='Closing price of Amazon')
plt.legend()
plt.title('Plot of dependent variable(closing price) versus no of sample(time)')
plt.xlabel('Date',fontsize=12)
plt.ylabel("Closing price",fontsize=12)
plt.show()

#5.b. ACF/PACF of the dependent variable.
def plot_acf_pacf(y, title):
    lags=60
    acf =sm.tsa.stattools.acf(y, nlags=lags)
    pacf =sm.tsa.stattools.acf(y, nlags=lags)

    fig = plt.figure(figsize = (8, 6))
    plt.subplot(211)
    plot_acf(y,ax=plt.gca(), lags =lags)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title('Autocorrelation of Amazon stock price ' +title, fontsize=14)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title('Partial Autocorrelation of Amazon stock price' +title, fontsize=14)
    fig.tight_layout(pad = 3)
    plt.show()

plot_acf_pacf(Amzn_df['Close'], title ='')


#5.c. Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient.

corr_mx =  Amzn_df.corr()
ax = sns.heatmap(corr_mx, vmin=-1, vmax = 1, center=0, cmap=sns.diverging_palette(20,220,n=200), square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation =45, horizontalalignment ='right')
ax.set_title("Heatmap of the matrix showing co-relation")
plt.show()

#5.d. Preprocessing procedures (if applicable): Clean the dataset (no missing data or NAN)
#checking if the dataset has any nan values

print(Amzn_df.isna().sum())

#using pearsons correlation
pearsoncorr = Amzn_df.corr(method='pearson')
print("The pearson correlation coefficeint is : \n",pearsoncorr)


#5.e.Split the dataset into train set (80%) and test set (20%).
Amzn_df_train, Amzn_df_test= train_test_split(Amzn_df, test_size = 0.2, shuffle=False)

X= Amzn_df[['Open','High','Low','Volume','Adj Close']]
Y =Amzn_df['Close']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, shuffle=False)

#6. Stationarity check


def plot_rolling_mean_var(x, title1, title2):

    mean_list_rolling_mean= []
    var_list_rolling_var= []

    for i in range(len(x)):
        try:
            each_row = x.head(i)
            mean_each_val = each_row.mean()
            var_each_row = each_row.var()
        except:
            mean_each_val = np.mean(x[:i + 1])
            var_each_row = np.var(x[:i + 1])

        mean_list_rolling_mean.append(mean_each_val)
        var_list_rolling_var.append(var_each_row)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(mean_list_rolling_mean)
    ax1.set_title(title1)
    ax1.set_xlabel("Observations")
    ax1.set_ylabel("Mean")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(var_list_rolling_var)
    ax2.set_title(title2)
    ax2.set_xlabel("Observations")
    ax2.set_ylabel("Variance")

    fig.tight_layout(pad =2)
    plt.show()

plot_rolling_mean_var(Amzn_df['Close'], 'Rolling mean of Amzn before differencing', 'Rolling variance of Amzn before differencing')



#ADF test
def ADF_Cal(x):
    result = adfuller(x)

    print("ADF Statistic: %f" %result[0])
    print("P-value: %f" %result[1])
    print('Critical values: ')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key,value))

ADF_Cal(Amzn_df['Close']) #caliing ADF function


#using differencing technique

log_transformed_data = np.log(Amzn_df['Close'])
plot_rolling_mean_var(log_transformed_data, 'Rolling mean of Amazon after log differencing', 'Rolling variance of Amazon after log differencing')
ADF_Cal(log_transformed_data)
plot_acf_pacf(log_transformed_data, title ='after LOG differencing')


print("AFTER LOG differening used first differencing")

#Amzn_diff1 = Amzn_df['Close'].diff()
Amzn_diff1 = log_transformed_data.diff()[1:]

ADF_Cal(Amzn_diff1)

plot_rolling_mean_var(Amzn_diff1, 'Rolling mean of Amazon using log and first differencing', 'Rolling variance of Amazon using log and first differencing')

plot_acf_pacf(Amzn_diff1, title='after log and first differencing')


#7.Time Series Decomposition

#using orginal y value data series for time series decomposition
df = pd.Series(np.array(Amzn_df['Close']), index = pd.date_range(start = '2000-01-03', periods =len(Amzn_df['Close']), freq='b'), name= 'Amazon Closing Price observation')

STL = STL(df)
res = STL.fit()

T=res.trend
S = res.seasonal
R = res.resid

#print(S)
#print(T)

fig = res.plot()
plt.show()

detrended = df -T  #detrended
adjusted_seasonal = df -S  #adjusted seasonal dataset


plt.figure()
#plt.plot(adjusted_seasonal, label='Adjusted Seasonal') #plots first 100 sample
plt.plot(detrended, label='Detrended dataset')
plt.plot(df, label ='Original set')
plt.legend()
plt.ylabel("Closing price")
plt.xlabel("Year")
plt.title("Plot of Detrended and original dataset ")
plt.show()

plt.figure()
plt.plot(adjusted_seasonal, label='Adjusted Seasonal')
#plt.plot(detrended, label='Detrended dataset')
plt.plot(df, label ='Original set')
plt.legend()
plt.ylabel("Closing price")
plt.xlabel("Year")
plt.title("Plot of Adjusted seasonal and original dataset ")
plt.show()



#----------------
#strength of Trend
#-------------
R= np.array(R)
T= np.array(T)
S= np.array(S)

#ratio_strength_trend = np.mac([0,1 - R.var()/adjusted_seasonal.var()])
ratio_strength_trend =np.max([0,1 - (R.var())/((T+R).var())])
print("\n\nThe strength of trend for this data set is ", ratio_strength_trend)


#----------------
#strength of seasonality
#-------------

ratio_strength_seasonlaity =np.max([0,1 -(R.var())/(S+R).var()])
print("\nThe strength of seasonality for this data set is", ratio_strength_seasonlaity)

# 8. Using the Holt-Winters method try to find the best fit

#print(Amzn_df_train)
#print(Amzn_df_test)

train_HLWM = ets.ExponentialSmoothing(Amzn_df_train['Close'],trend='multiplicative',damped_trend=False,seasonal='multiplicative', seasonal_periods =12).fit()
HLWM_prediction_train = train_HLWM.forecast(steps=len(Amzn_df_train['Close']))
test_HLWM = train_HLWM.forecast(steps=len(Amzn_df_test['Close']))
test_predict_HLWM = pd.DataFrame(test_HLWM).set_index(Amzn_df_test['Close'].index)

resid_HLWM = np.subtract(Y_train.values,np.array(HLWM_prediction_train))
forecast_error_HLWM = np.subtract(Y_test.values,np.array(test_HLWM))

MSE_HLWM = np.square(resid_HLWM).mean()
print("Mean square error for (training set) HLWM is ", MSE_HLWM)

#print(test_predict_HLWM)


def auto_corelation(x, bar_x, n):
    list_ry =[]
    for i in range(n):
        total_numerator = 0
        total_dinom = 0
        a = i
        for t in range(len(x)):
            if a < len(x):
                each_sum = (x[a] - bar_x) * (x[t] -bar_x)
                total_numerator = each_sum + total_numerator
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
                a = a +1
            else:
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
        value = total_numerator / total_dinom
        list_ry.append(value)

    inverse_generated_list_ry = list_ry[::-1]
    Generated_list_Ry = inverse_generated_list_ry[:-1] + list_ry
    Generated_list_Ry_np = np.array(Generated_list_Ry)
    #return list_ry, Generated_list_Ry_np

    return Generated_list_Ry_np


# ================== Auto-correlation function=============

def calc_auto_corelation(x, bar_x, n, title):
    list_ry = []
    for i in range(n):
        total_numerator = 0
        total_dinom = 0
        a = i
        for t in range(len(x)):
            if a < len(x):
                each_sum = (x[a] - bar_x) * (x[t] - bar_x)
                total_numerator = each_sum + total_numerator
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
                a = a + 1
            else:
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
        value = total_numerator / total_dinom
        list_ry.append(value)

    inverse_generated_list_ry = list_ry[::-1]
    Generated_list_Ry = inverse_generated_list_ry[:-1] + list_ry
    Generated_list_Ry_np = np.array(Generated_list_Ry)

    x_np = np.linspace(-19, 19, 39)
    n = 1.96/(np.sqrt(len(Generated_list_Ry_np)))
    plt.axhspan(-n,n, alpha = 0.25, color= 'blue')
    plt.stem(x_np, Generated_list_Ry_np)
    plt.xlabel("Lags", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.title(title, fontsize=14)
    plt.show()

    return list_ry, Generated_list_Ry_np



list_acf, acf_resid = calc_auto_corelation(resid_HLWM, np.mean(resid_HLWM), 20, title ='ACF plot of residuals for Holt-Winter method')


Q  = (len(resid_HLWM)) *np.sum(np.square(acf_resid)[21:])

print("\nThe Q value of residual using HWM is  ",Q)

print(f"The Mean of residual of HLWM is {np.mean(resid_HLWM)}")
print(f"The variance of residual of HLWM is {np.var(resid_HLWM)}")

MSE = np.square(np.subtract(Amzn_df_test['Close'].values,np.ndarray.flatten(test_HLWM.values))).mean()
print("Mean square error for Holt Winter method is of testing set is ", MSE)


print(f"The Mean of forecast of HLWM is {np.mean(forecast_error_HLWM)}")
print(f"The variance of forecast of HLWM is {np.var(forecast_error_HLWM)}")

print(f"\n The ratio of resid vs forecast is {(np.var(resid_HLWM)) / (  np.var(forecast_error_HLWM)  )}")

plt.plot(Amzn_df_train['Close'],label= "Amazon-train")
plt.plot(Amzn_df_test['Close'],label= "Amazon-test")
plt.plot(test_predict_HLWM,label= "Holt-Winter Method-test")
plt.legend(loc='upper left')
plt.title('Holt-Winter Method')
plt.xlabel('Time (daily)')
plt.ylabel('Closing Price')
plt.show()

#============
# Feature selection
#==========================
X= Amzn_df[['Open','High','Low','Volume','Adj Close']]
Y =Amzn_df['Close']

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, shuffle=False)
X_train_copy = X_train.copy()
X_test_copy = X_test

X_val = np.c_[np.ones((4258,1)),X_train.values]
X_val_cond = np.c_[X_train.values]
H_vector = np.matmul(X_val.T, X_val)
s, d, v = np.linalg.svd(H_vector)
print(f"SingularValues = {d}")
print(f"\nThe condition number including constant in original data= {LA.cond(X_val)}")
print(f"The condition number without constant (original data) = {LA.cond(X_val_cond)}")


X_train=sm.add_constant(X_train)
model = sm.OLS(Y_train, X_train).fit()
X_test = sm.add_constant(X_test)
predictions =model.predict(X_test)
print('')
print(model.summary())

#Using backstep regression

###--------Removing high

print("\n\n\nRemoving High")
X_train.drop(['High'], axis =1, inplace =True)
model = sm.OLS(Y_train, X_train).fit()
X_test.drop(['High'], axis =1, inplace =True)
predictions1 =model.predict(X_test)
print(model.summary())

###--------Removing const

print("\n\n\nRemoving const")
X_train.drop(['const'], axis =1, inplace =True)
model = sm.OLS(Y_train, X_train).fit()
X_test.drop(['const'], axis =1, inplace =True)
predictions2 =model.predict(X_test)
print(model.summary())

###--------Removing low

print("\n\n\nRemoving low")
X_train.drop(['Low'], axis =1, inplace =True)
model = sm.OLS(Y_train, X_train).fit()
X_test.drop(['Low'], axis =1, inplace =True)
predictions3 =model.predict(X_test)
print(model.summary())

###--------Removing Open

print("\n\n\nRemoving Open")
X_train.drop(['Open'], axis =1, inplace =True)
model = sm.OLS(Y_train, X_train).fit()
X_test.drop(['Open'], axis =1, inplace =True)
predictions4 =model.predict(X_test)
print(model.summary())

###--------Removing Adj Close will decrease adjusted r squared

# prediction 4 is the final prediction for x test based on my multiple regression model
#so I will name it as prediction_test

prediction_test = predictions4


#ftest --- Multiple regression
f_test = model.fvalue
print('\nF-statistic: ', f_test)

print("Probability of observing value at least as high as F-statistic ",model.f_pvalue)

#t-test
t_Test =model.pvalues
print("\nT-test P values : ", t_Test)

#now I nedd to predict based on train set so....

model = sm.OLS(Y_train, X_train).fit()
prediction_train =model.predict(X_train)
training_residual = np.subtract(Y_train,prediction_train)

MSE = np.square(training_residual).mean()
print("Mean square error of residuals for multiple regression is ", MSE)
print(f"RMSE for training set using multiple regression is :{np.sqrt(MSE)} ")



def calc_Q_value(x):
    #calc_autocorr,calc_autocorr_np = auto_corelation(x, statistics.mean(x), n = 5)
    Q_calc_autocorr = []
    for i in x:
        i = i ** 2
        Q_calc_autocorr.append(i)
    Q_value = len(x) * sum(Q_calc_autocorr[21:])
    return Q_value



#calling auto-corelation function
list_acf_residauls, np_acf_calc_residuals = calc_auto_corelation(training_residual, np.mean(training_residual), n=20, title='Plot of ACF of residuals of regression')


#calling function to calculate Q values

Q  = (len(training_residual)) *(np.sum(np.square(np_acf_calc_residuals)[21:]))

print(f"The Q value of residual of regression is {Q}")

print(f"The mean of residuals is {np.mean(training_residual)}")
print(f"The variance of residual is {np.var(training_residual)}")



testing_error_regression= np.subtract(Y_test, prediction_test)
MSE = np.square(testing_error_regression).mean()
print("\nMean square error for testing set multiple regression is ", MSE)
print(f"RMSE for testing set using multiple regression is :{np.sqrt(MSE)} ")


print(f"The mean of forecast of multiple regression is {np.mean(testing_error_regression)}")
print(f"The variance of foreacast of multiple regression is {np.var(testing_error_regression)}")

print(f"\n The ratio of resid vs forecast is {np.var(training_residual)/np.var(testing_error_regression)}")


#plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(prediction_test, label = ' one-step prediction using multiple regression method')
plt.xlabel("Date (year)", fontsize= 14)
plt.ylabel("Closing price", fontsize =14)
plt.title("Plot of Amzn stock price using regression method", fontsize =14)
plt.legend()
plt.show()

#14. Base models

#Averge method

y_predict_train_set = []
value = 0
for i in range(len(Y_train)):
    if i != 0:
        value = value + Y_train[i - 1]
        t_value  = i
        y_each_predict = value / i
        y_predict_train_set.append(y_each_predict)
    else:
        continue

#print(y_predict_train_set)



y_predict_test_set= []
for i in range(len(Y_test)):
    y_predict_each = sum(Y_train) / len(Y_train)
    y_predict_test_set.append(y_predict_each)

y_preidction_average= pd.DataFrame(y_predict_test_set).set_index(Y_test.index)

#print(y_predict_test_set)

plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(y_preidction_average, label = 'forecast using average method')
plt.xlabel("Date (year)", fontsize= 14)
plt.ylabel("Closing price", fontsize =14)
plt.title("Plot of Amzn stock price using average method", fontsize =14)
plt.legend()
plt.show()

#now lets find out the MSE of our prediction error --on training set using average method

error_train_set_avg = np.subtract(Y_train[1:], y_predict_train_set)
#print(error_train_set_avg)

def calc_MSE(x):
    MSE = np.square(np.array(x)).mean()
    return MSE


MSE_train_set =calc_MSE(error_train_set_avg)
#MSE_train_set = np.sum((var_square_train_set_array) ** 2 )/ len(t_train_set)
print(f"\nMSE of prediction error (training set) using average method is : {MSE_train_set}")

#calling auto-corelation function
list_acf_residauls_average, np_acf_calc_residuals_average = calc_auto_corelation(error_train_set_avg, np.mean(error_train_set_avg), n=20, title='Plot of ACF of residuals using average method')

#calling function to calculate Q values

Q_residual  = (len(error_train_set_avg)) *(np.sum(np.square(np_acf_calc_residuals_average)[21:]))

print(f"The Q value of residual using average method is {Q_residual}")



print(f"The mean of residuals using average method is {np.mean(error_train_set_avg)}")
print(f"The variaince of residual using average method is {np.var(error_train_set_avg)}")

error_test_set_avg = np.subtract(Y_test, y_predict_test_set)
MSE_train_set =calc_MSE(error_test_set_avg)
print(f"\nMSE of forecast (testing set) using average method is : {MSE_train_set}")
print(f"Mean of forecast error is: {np.mean(np.array(error_test_set_avg))}")
print(f"Variance of forecast error is: {np.var(np.array(error_test_set_avg))}")


print(f"\n The ratio of resid vs forecast of average method is {( np.var(error_train_set_avg) ) / (  np.var(np.array(error_test_set_avg))  )}")



#Naive method

print("****  Naive Method   ****")
y_predict_train_set_naive = []
value = 0
for i in range(len(Y_train[1:])):
    y_predict_train_set_naive.append(Y_train[i])

#print(y_predict_train_set_naive)

y_predict_test_set_naive= [Y_train[-1] for i in Y_test]

y_prediction_naive_test= pd.DataFrame(y_predict_test_set_naive).set_index(Y_test.index)

#print(y_predict_test_set_naive)

plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(y_prediction_naive_test, label = ' forecast using naive method')
plt.xlabel("Date (Year)", fontsize= 14)
plt.ylabel("Closing Price", fontsize =14)
plt.title("Plot of Amzn stock price using naive method", fontsize =14)
plt.legend()
plt.show()

error_train_set_naive = np.subtract(Y_train[1:], y_predict_train_set_naive)

error_test_set_naive = np.subtract(Y_test, y_predict_test_set_naive)

MSE_train_set_naive =calc_MSE(error_train_set_naive)
print(f"\nMSE of prediction error (training set) using naive method is : {MSE_train_set_naive}")




#calling auto-corelation function
list_acf_residauls_naive, np_acf_calc_residuals_naive = calc_auto_corelation(error_train_set_naive, np.mean(error_train_set_naive), n=20, title='Plot of ACF of residuals using naive method')

#calling function to calculate Q values

Q_residual  = (len(error_train_set_naive)) *(np.sum(np.square(np_acf_calc_residuals_naive)[21:]))
print(f"The Q value of residual using naive method is {Q_residual}")



print(f"The mean of residuals using naive method is {np.mean(error_train_set_naive)}")
print(f"The variance of residual using naive method is {np.var(error_train_set_naive)}")



MSE_test_set_naive =calc_MSE(error_test_set_naive)
print(f"\nMSE of prediction error (testing set) using naive method is : {MSE_test_set_naive}")

print(f"Mean of forecast error using naive method is: {np.mean(np.array(error_test_set_naive))}")
print(f"Variance of forecast error using naive method is: {np.var(np.array(error_test_set_naive))}")


print(f"\n The ratio of resid vs forecast of naive method is {( np.var(error_train_set_naive) ) / ( np.var(np.array(error_test_set_naive)) )}")



#Drift method

print("***** Drift Method   ********")

y_predict_train_set_drift = []
value = 0
for i in range(len(Y_train)):
    if i > 1:
        slope_val = (Y_train[i - 1] - Y_train[0]) / (i-1)
        y_each_predict = (slope_val * i) + Y_train[0]
        y_predict_train_set_drift.append(y_each_predict)
    else:
        continue

y_predict_test_set_drift= []
for h in range(len(Y_test)):
    slope_val = (Y_train[-1] - Y_train[0] ) /( len(Y_train) - 1 )
    y_predict_each = Y_train[-1] + ((h +1) * slope_val)
    y_predict_test_set_drift.append(y_predict_each)

#print(y_predict_test_set_drift)

y_preidction_drift= pd.DataFrame(y_predict_test_set_drift).set_index(Y_test.index)

plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(y_preidction_drift, label = 'forecast using drift method')
plt.xlabel("Date (Year)", fontsize= 14)
plt.ylabel("Closing Price", fontsize =14)
plt.title("Plot of Amzn stock price using drift method", fontsize =14)
plt.legend()
plt.show()

error_train_set_drift = np.subtract(Y_train[2:], y_predict_train_set_drift)

error_test_set_drift = np.subtract(Y_test, y_predict_test_set_drift)

MSE_train_set_drift =calc_MSE(error_train_set_drift)
print(f"\nMSE of prediction error (training set) using drift method is : {MSE_train_set_drift}")



#calling auto-corelation function
list_acf_residauls_drift, np_acf_calc_residuals_drift = calc_auto_corelation(error_train_set_drift, np.mean(error_train_set_drift), n=20, title='Plot of ACF of residuals using drift method')

#calling function to calculate Q values

Q_residual  = (len(error_train_set_drift)) *(np.sum(np.square(np_acf_calc_residuals_drift)[21:]))

print(f"The Q value of residual using drift method is {Q_residual}")


print(f"The mean of residuals using drift method is {np.mean(error_train_set_drift)}")
print(f"The variance of residual using drift method is {np.var(error_train_set_drift)}")


MSE_test_set_drift =calc_MSE(error_test_set_drift)
print(f"\nMSE of prediction error of testing set using drift method is : {MSE_test_set_drift}")



print(f"Mean of forecast error using drift method is: {np.mean(np.array(error_test_set_drift))}")
print(f"Variance of forecast error using drift method is: {np.var(np.array(error_test_set_drift))}")


print(f"\n The ratio of resid vs forecast of drift method is {( np.var(error_train_set_drift) ) / ( np.var(np.array(error_test_set_drift)) )}")

#Simple and exponential smoothing

print("*******  SES Method *****")

SES = ets.ExponentialSmoothing(Y_train,trend=None,damped=False,seasonal=None).fit()
SES_predict_train= SES.forecast(steps=len(Y_train))
SES_predict_test= SES.forecast(steps=len(Y_test))
predict_test_SES = pd.DataFrame(SES_predict_test).set_index(Y_test.index)

resid_SES = np.subtract(Y_train.values,np.array(SES_predict_train))
forecast_error_Ses = np.subtract(Y_test.values,np.array(SES_predict_test))

MSE_SES = np.square(resid_SES).mean()
print("Mean square error for (training set) simple exponential smoothing is ", MSE_SES)

plt.plot(Y_train,label= "Amazon-train")
plt.plot(Y_test,label= "Amazon-test")
plt.plot(predict_test_SES,label= "SES Method prediction")
plt.legend(loc='upper left')
plt.title('SES method')
plt.xlabel('Time (daily)')
plt.ylabel('Closing Price')
plt.show()


#calling auto-corelation function
list_acf_residauls_SES, np_acf_calc_residuals_SES = calc_auto_corelation(resid_SES, np.mean(resid_SES), n=20, title='Plot of ACF of residuals using SES method')

#calling function to calculate Q values

Q_residual  = (len(resid_SES)) *(np.sum(np.square(np_acf_calc_residuals_SES)[21:]))

print(f"The Q value of residual using SES method is {Q_residual}")


print(f"Mean of residual using SES method is: {np.mean(np.array(resid_SES))}")
print(f"Variance of residual using SES method is: {np.var(np.array(resid_SES))}")


MSE_SES = np.square(forecast_error_Ses).mean()
print("Mean square error for (testing set) simple exponential smoothing is ", MSE_SES)


print(f"Mean of forecast error using SES method is: {np.mean(np.array(forecast_error_Ses))}")
print(f"Variance of forecast error using SES method is: {np.var(np.array(forecast_error_Ses))}")

print(f"\n The ratio of resid vs forecast of SES method is {( np.var(np.array(resid_SES)) ) / ( np.var(np.array(forecast_error_Ses)) )}")



def auto_corelation(x, bar_x, n):
    list_ry =[]
    for i in range(n):
        total_numerator = 0
        total_dinom = 0
        a = i
        for t in range(len(x)):
            if a < len(x):
                each_sum = (x[a] - bar_x) * (x[t] -bar_x)
                total_numerator = each_sum + total_numerator
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
                a = a +1
            else:
                each_dinom = (x[t] - bar_x) ** 2
                total_dinom = each_dinom + total_dinom
        value = total_numerator / total_dinom
        list_ry.append(value)

    inverse_generated_list_ry = list_ry[::-1]
    Generated_list_Ry = inverse_generated_list_ry[:-1] + list_ry
    Generated_list_Ry_np = np.array(Generated_list_Ry)
    #return list_ry, Generated_list_Ry_np

    return Generated_list_Ry_np



AR_ACF_array = auto_corelation(Amzn_diff1, np.mean(Amzn_diff1), 20)


lags =20
Ry = AR_ACF_array

####=====================
# ======= calculating GPAC table
##======================
def calc_Gpac(na_order, nb_order, Ry):
    x = int((len(Ry) - 1) / 2)
    df = pd.DataFrame(np.zeros((na_order, nb_order + 1)))
    df = df.drop(0, axis=1)

    for k in df:  # this for loop iterates over the column to calculate the value
        for j, row_val in df.iterrows():  # this iterates over the rows
            if k == 1:  # for first column
                dinom_val = Ry[x + j]  # Here Ry(0) = lags -1   = x
                numer_val = Ry[x + j + k]
            else:  # when our column is 2 or more than 2; when k > 2
                dinom_matrix = []
                for rows in range(k):  # this loop is for calculating the square matrix (iterating over the rows of matrix)
                    # print(rows)
                    row_list = []
                    for col in range(k):  # this loop is for calculating the square matrix (iterating over the columns of matrix)
                        # print(col)
                        each = Ry[x - col + rows + j]
                        # print(each)
                        row_list.append(each)
                    dinom_matrix.append(np.array(row_list))

                # dinominator matrix and numerator matrix have same values except for the last column so:
                dinomator_matrix = np.array(dinom_matrix)
                numerator_matrix = np.array(dinom_matrix)

                # updating values for last column of numerator matrix

                last_col =k
                for r in range(k):
                    numerator_matrix[r][last_col - 1] = Ry[x + r + 1 + j]

                # calculating determinants
                numer_val = np.linalg.det(numerator_matrix)
                dinom_val = np.linalg.det(dinomator_matrix)

            df[k][j] = numer_val / dinom_val  # plugs the value in GPAC table

    print(df)

    import seaborn as sns
    sns.heatmap(df, cmap=sns.diverging_palette(20, 220, n=200), annot=True, center=0)
    plt.title('Generalized Partial Auto-correlation Table')
    plt.show()


na_order =8
nb_order =8
calc_Gpac(na_order, nb_order, Ry)

y =Amzn_df['Close']
#mean_y = y.mean()
#y = y - y.mean()

Y_train, Y_test = train_test_split(y, test_size = 0.2, shuffle=False)

def calc_theta(y,na,nb, theta):
    if na == 0:
        dinominator = [1]
    else:
        dinominator =np.append([1], theta[:na])

    if nb ==0:
        numerator = [1]
    else:
        numerator = np.append([1], theta[-nb:])

    diff = na -nb
    if diff > 0:
        numerator =np.append(numerator, np.zeros(diff))


    sys = (dinominator, numerator, 1)
    _,e = signal.dlsim(sys, y)
    theta =[]
    for i in e:
        theta.append(i[0])

    theta_e =np.array(theta)

    return theta_e

def step_1(y, na, nb, theta, delta):
    e = calc_theta(y,na,nb, theta)
    SSE = np.dot(e, e.T)

    X=[]

    n =na +nb

    for i in range(n):
        theta_new =theta.copy()
        theta_new[i] =theta_new[i] +delta
        new_e = calc_theta(y, na, nb, theta_new)
        x_i = np.subtract(e, new_e)/delta
        X.append(x_i)


    X =  np.transpose(X)
    A =  np.transpose(X).dot(X)
    g=  np.transpose(X).dot(e)

    return A,g, SSE

def step_2(theta, A, g, u, y):
    n = na +nb
    idt = np.identity(n)
    before_inv= A + (u * idt)
    AUI_inv = np.linalg.inv(before_inv)
    diff_theta= AUI_inv.dot(g)
    theta_new = theta +diff_theta

    new_e = calc_theta(y, na, nb, theta_new)
    SSE_new = new_e.dot(new_e.T)
    return SSE_new, theta_new, diff_theta

def calc_LMA(count, y, na, nb, theta, delta, u, u_max):
    i =0
    SSE_count=[]
    norm_theta=[]

    while i <count:
        A,g,SSE = step_1(y,na,nb,theta, delta)
        SSE_new, theta_new, diff_theta =step_2(theta, A, g,u, y)
        SSE_count.append(SSE_new)

        n =na+nb

        if SSE_new < SSE:
            norm_theta2  = np.linalg.norm(np.array(diff_theta),2)
            norm_theta.append(norm_theta2)

            if norm_theta2 < 0.001:
                theta = theta_new.copy()
                break

            else:
                theta =theta_new.copy()
                u = u/10

        while SSE_new >= SSE:
            u = u *10
            if u>u_max:
                print("Mue is high now and cannot go higher than that!!!")
                break
            SSE_new, theta_new, diff_theta = step_2(theta, A, g, u, y)
        theta  = theta_new
        i += 1

    variance_error = SSE_new / (len(y) - n)
    co_variance =variance_error * np.linalg.inv(A)
    print("The estimated parameters >>> ", theta)
    print(f"\n The estimated co-variance matrix is {co_variance}")
    print(f"\n The estimated variance of error is {variance_error}")


    for i in range(na):
        std_deviation = np.sqrt(co_variance[i][i])
        print(f"The standard deviation for a{i+1} is {std_deviation}")

    for j in range(na, n):
        std_deviation = np.sqrt(co_variance[j][j])
        print(f"The standard deviation for b{i + 1} is {std_deviation}")

    print(f"\n The confidence interval for parameters are: ")
    for i in range(na):
        interval = 2 * np.sqrt(co_variance[i][i])
        print(f"{(theta[i]- interval)} < a{i+1} < {(theta[i] + interval)}")

    for j in range(na,n):
        interval = 2 * np.sqrt(co_variance[j][j])
        print(f"{(theta[j]- interval)} < b{j -na + 1} < {(theta[j] + interval)}")

    #zero/pole

    num_root =[1]
    den_root= [1]

    for i in range(na):
        num_root.append(theta[i])
    for i in range(nb):
        den_root.append(theta[i + na])

    poles =np.roots(num_root)
    zeros = np.roots(den_root)

    print(f"\nThe roots of the numerators are {zeros}")
    print(f"The roots of dinominators are {poles}")

    #plt.plot(SSE_count)
    #plt.xlabel("Numbers of Iternations")
    #plt.ylabel("Sum square Error")
    #plt.title("Sum of square Error vs the iterations")
    #plt.show()

    return theta,co_variance


delta = 10**-6
na =2
nb =0
n = na +nb
theta = np.zeros(n)
u = 0.01
u_max = 1e10
count = 60

print("\nFor our estimated ARMA (2,O): ")

theta, cov = calc_LMA(count, Amzn_df['Close'], na,nb, theta, delta, u, u_max)
Y_train , Y_test = train_test_split(Amzn_df['Close'], test_size= 0.2, shuffle =False)

#y_prediction = one_step_pred(theta,na,nb,Y_train)
#y_prediction = one_step_pred(theta,na,nb,Y_train)    -0.9421188128435487
y_predict=[]
for i in range(len(Y_train)):
    if i ==0:
        predict = (-theta[0]) * Y_train[i]
    else:
        predict = ( - theta[0] * Y_train[i] ) + ( -theta[1] * Y_train[i-1])
    y_predict.append(predict)

resid = np.subtract(np.array(Y_train),np.array(y_predict))
acf_resid = auto_corelation(resid, np.mean(resid), 20)


Q  = (len(resid)) *(np.sum(np.square(acf_resid)[21:]))
DOF = len(resid) - na -nb
alfa =0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("\nThe Q value is  ",Q)

print(f"The chi-critical is {chi_critical}")


if Q <chi_critical:
    print(f"\n The Q value is less than {chi_critical} (chi-critical) so, the residual is white")
else:
    print(f"\nThe Q value is not less than {chi_critical} (chi-critical) so, the residual is not white")

print(resid[:5])

print(f"Mean of residual with ARMA(2,0) is {np.mean(resid)}")
print(f"The variance of residual with ARMA(2,0) is {np.var(resid)}")


y_predict = pd.DataFrame(y_predict).set_index(Y_train.index)

plt.plot(Y_train, label='Y_train')
plt.plot(y_predict, label ='Predicted values')
plt.xlabel('Number of observations')
plt.ylabel('y-values')
plt.title("One-step-ahead prediction")
plt.legend()
plt.show()


print("\nFor our estimated ARMA (3,O): ")
#delta = 10**-6
na =3
nb =0
n = na +nb
theta = np.zeros(n)
#u = 0.01
#u_max = 1e10
#count = 60


theta, cov = calc_LMA(count, Amzn_df['Close'], na,nb, theta, delta, u, u_max)
Y_train , Y_test = train_test_split(Amzn_df['Close'], test_size= 0.2, shuffle =False)


print(f"The estimated parameters are {theta[0]} and {theta[1]}")
na  = 2
print("So lets do one-step prediction accordingly.")
#y_prediction = one_step_pred(theta,na,nb,Y_train)
y_predict=[]
for i in range(len(Y_train)):
    if i ==0:
        predict = (-theta[0]) * Y_train[i]
    else:
        predict = ( - theta[0] * Y_train[i] ) + ( -theta[1] * Y_train[i-1])
    y_predict.append(predict)

resid = np.subtract(np.array(Y_train),np.array(y_predict))
acf_resid = auto_corelation(resid, np.mean(resid), 20)

print(resid[:5])


Q  = (len(resid)) *np.sum(np.square(acf_resid)[21:])
DOF = len(resid) - na -nb
alfa =0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("\nThe Q value is  ",Q)

print(f"The chi-critical is {chi_critical}")


if Q <chi_critical:
    print(f"\n The Q value is less than {chi_critical} (chi-critical) so, the residual is white")
else:
    print(f"\nThe Q value is not less than {chi_critical} (chi-critical) so, the residual is not white")

print(f"MSE of residual for ARMA(2,0) is {np.square(resid).mean()}")
print(f"\nMean of residual with ARMA(2,0) is {np.mean(resid)}")
print(f"The variance of residual with ARMA(2,0) is {np.var(resid)}")

y_predict = pd.DataFrame(y_predict).set_index(Y_train.index)

plt.plot(Y_train, label='Y_train')
plt.plot(y_predict, label ='Predicted values')
plt.xlabel('Number of observations')
plt.ylabel('y-values')
plt.title("One-step-ahead prediction")
plt.legend()
plt.show()


# prediction for test set
y_prediction_test=[]
for i in range(len(Y_test)):
    if i ==0:
        predict = (-theta[0] * Y_train[-1])+ (-theta[1] * Y_train[-2])
    elif i ==1:
        predict = ( -theta[0] * y_prediction_test[0] ) + (-theta[1] * Y_train[-1])
    elif i ==2:
        predict = ( -theta[0] * y_prediction_test[1] ) + (-theta[1] * y_prediction_test[0])
    else:
        predict = (-theta[0] * y_prediction_test[i - 1]) + (-theta[1] * y_prediction_test[i-2])
    y_prediction_test.append(predict)

y_prediction_test = pd.DataFrame(y_prediction_test).set_index(Y_test.index)

forecast_error = np.subtract(np.array(Y_test), np.array(y_prediction_test))

print(f"\nMSE of forecast for ARMA(2,0) is {np.square(forecast_error).mean()}")
print(f"The mean of trainings set error is {np.mean(forecast_error)}")
print(f"The variance of training set error is {np.var(forecast_error)}")

ratio = np.var(resid)/np.var(forecast_error)

print(f"\nThe ratio of variance of residual to variance of forecast is {ratio}")

plt.plot(Y_train, label='Training set')
plt.plot(Y_test, label = 'Testing set')
plt.plot(y_prediction_test, label ='Prediction Test set')
plt.xlabel("Date (Year)", fontsize= 14)
plt.ylabel("Closing Price", fontsize =14)
plt.title("Plot of Amzn stock price using ARMA(2,0) ", fontsize =14)
plt.legend()
plt.show()

#===================ARIMA MODEL ==================================

delta = 10**-6
na =2
nb =0
n = na +nb
theta = np.zeros(n)
u = 0.01
u_max = 1e10
count = 60

print(f"For our ARIMA(2,0)")

theta, cov = calc_LMA(count, Amzn_diff1, na,nb, theta, delta, u, u_max)



# this amazon diff is in different domain so we make ckomparison with the same domain so lets divid Amazon diff1 to training and testing
thet = 0.04469821 # we only got one parameter from LM
na = 1
nb =0

print(f"The estimated parameter after removing the statistically not significant parameter is {thet}")

# ========== Writing the model==================
#   ARIMA (1,1,0)   ---- (1 + 0.04469821 q **-1) (1- q**-1 ) y(t)  = e(t)
#                 ---- y(t) - q **-1 y(t) + 0.04469821 q **-1 y(t)  - 0.04469821 q **-2 y(t)  = e(t)
#                      y(t) = 0.95530179 q **-1 y(t) - 0.04469821 q **-2 y(t) + e(t)
#                     y_hat_(t) = 0.95530179 y(t-1) - 0.04469821 y(t-2) + e(t)

Y_train, Y_test = train_test_split(Amzn_df['Close'], test_size=0.2, shuffle =False)

diff_Y_train, diff_Y_test = train_test_split(Amzn_diff1, test_size = 0.2, shuffle=False)

#y_prediction_diff = one_step_pred(thet ,na,nb,Y_train)

#======= One-step prediction ==========================

y_predict_hat=[]
for i in range(len(Y_train)):
    if i ==0:
        predict = 0.95530179 * Y_train[i]
    else:
        predict = ( 0.95530179 * Y_train[i] ) - (0.04469821 * Y_train[i-1])
    y_predict_hat.append(predict)


# Residual calculation


def calc_MSE(x):
    MSE = np.square(np.array(x)).mean()
    return MSE

resid = np.subtract(Y_train,y_predict_hat)
#resid = np.subtract(diff_Y_train,y_predict_diff)\
MSE_resid = calc_MSE(resid)

print(f"MSE of resid of ARIMA(1,1,0) is {MSE_resid}")
acf_resid = auto_corelation(resid, np.mean(resid), 20)


Q  = (len(resid)) *np.sum(np.square(acf_resid)[21:])
DOF = len(resid)- na -nb
alfa =0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("\nThe Q value is  ",Q)


print(f"The chi-critical is {chi_critical}")


if Q <chi_critical:
    print(f"\n The Q value is less than {chi_critical} (chi-critical) so, the residual is white")
else:
    print(f"\nThe Q value is not less than {chi_critical} (chi-critical) so, the residual is not white")

print(f"Mean of residual with ARIMA(1,1,0) is {np.mean(resid)}")
print(f"The variance of residual with ARIMA(1,1,0) is {np.var(resid)}")


#========= One-step prediction plot ====================
#y_predict_diff = pd.DataFrame(y_predict_diff).set_index(diff_Y_train.index)
y_predict_hat = pd.DataFrame(y_predict_hat).set_index(Y_train.index)

plt.plot(Y_train, label='Original Y_train')
plt.plot(y_predict_hat, label ='Predicted values')
plt.xlabel('Time (Date-Month)')
plt.ylabel('Closing price ($)')
plt.title("One-step-ahead prediction for original plot using ")
plt.legend()
plt.show()

# Here back transformation is just the original dataset
#==========Back transformation

diff_transform=[]
for i in range(len(log_transformed_data)):
    if i ==0:
        val =log_transformed_data[i]
    else:
        val = Amzn_diff1[i-1] + log_transformed_data[i -1]

    diff_transform.append(val)

anti_log_transformation =[]
for k in diff_transform:
    anti_log_transformation.append(np.exp(k))

#  y_hat_(t) = 0.95530179 y(t-1) - 0.04469821 y(t-2) + e(t)
def forecast_func(thet, Y_train,step):

    y_prediction_test=[]
    for i in range(step):
        if i ==0:
            predict = (thet[0] * Y_train[-1])- (thet[1] * Y_train[-2])
        elif i ==1:
            predict = ( thet[0] * y_prediction_test[0] ) - (thet[1] * Y_train[-1])
        elif i ==2:
            predict = ( thet[0] * y_prediction_test[1] ) - (thet[1] * y_prediction_test[0])
        else:
            predict = (thet[0] * y_prediction_test[i - 1]) - (thet[1] * y_prediction_test[i-2])
        y_prediction_test.append(predict)

    return y_prediction_test

thet = [0.95530179 , 0.04469821]

y_prediction_test = forecast_func(thet, Y_train, step =len(Y_test))


y_prediction_test = pd.DataFrame(y_prediction_test).set_index(Y_test.index)

forecast_error = np.subtract(np.array(Y_test), np.array(y_prediction_test))

MSE_testing_error = calc_MSE(forecast_error)
print(f"MSE of prediction for testing error is {MSE_testing_error}")
print(f"\nThe mean of error of testing ARIMA is{np.mean(forecast_error)}")
print(f"The variance of forecast error is {np.var(forecast_error)}")
print(f"\nThe variance of forecast error is {np.var(forecast_error)}")

ratio = np.var(resid)/np.var(forecast_error)

print(f"The ratio of variance of residual to variance of forecast is {ratio}")


#=============================================
#========== Forecast function ==============
#===================================

print("****  Drift Method   ****")

def forecast_function(Y_train, step):
    y_predict_test_set_drift= []
    for h in range(step):
        slope_val = (Y_train[-1] - Y_train[0] ) /( len(Y_train) - 1 )
        y_predict_each = Y_train[-1] + ((h +1) * slope_val)
        y_predict_test_set_drift.append(y_predict_each)

    return y_predict_test_set_drift

step = len(Y_test)
y_predict_test_set_drift = forecast_function(Y_train, step)

y_predict_test_set_drift = pd.DataFrame(y_predict_test_set_drift).set_index(Y_test.index)



#print(y_predict_test_set_naive)

plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(y_predict_test_set_drift, label = ' forecast using drift method')
plt.xlabel("Date (Year)", fontsize= 14)
plt.ylabel("Closing Price", fontsize =14)
plt.title("Plot of Amzn stock price using drift method", fontsize =14)
plt.legend()
plt.show()

#============ H-step prediction =======================================


step =100
y_predict_h_step_drift = forecast_function(Y_train, step)

y_predict_h_step_drift= pd.DataFrame(y_predict_h_step_drift).set_index(Y_test[:100].index)

plt.plot(Y_test[:100], label = 'Test set')
plt.plot(y_predict_h_step_drift, label = ' 100 -step forecast using naive method')
plt.xlabel("Date (Year)", fontsize= 14)
plt.ylabel("Closing Price", fontsize =14)
plt.title("Plot of H-step prediction", fontsize =14)
plt.legend()
plt.show()

