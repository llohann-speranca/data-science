import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Statistical tools
from scipy.signal import periodogram
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.graphics.tsaplots import plot_pacf

# Linear Models

from sklearn.linear_model import LinearRegression


# Multioutput

from sklearn.multioutput import RegressorChain






plot_params = {'color': '0.75',
 'style': '.-',
 'markeredgecolor': '0.25',
 'markerfacecolor': '0.25',
 'legend': False}







def get_trend(df, y,constant=True, dp_order=1, is_seasonal=True, dp_drop=True,
                    model=LinearRegression(),
                    fourier=None,
                    dp=None,
                    **DeterministicProcesskwargs) -> pd.DataFrame:
    """ 
    Returns the trend of df. 
    Default parameters: window=365, center=False, min_periods=0
    """
    if dp is None:
        dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=dp_order,
        seasonal=is_seasonal,
        drop=dp_drop,
        **DeterministicProcesskwargs
        )
    
    X = dp.in_sample()
    model = LinearRegression().fit(X, y)
    trend = pd.DataFrame(
                        model.predict(X),
                        index=X.index
    )



    return trend

# A newer version is in BTC-periodogram.ipynb as get_season (need to revisit this notebook for the auto-forecaster project - to get seasons automatically)

def deseasonalize(df: pd.Series, season_freq='A', fourier_order=0, 
                    constant=True, dp_order=1,  dp_drop=True,
                    model=LinearRegression(),
                    fourier=None,
                    dp=None,
                    **DeterministicProcesskwargs):
    """
    Deaseasonalize and detrend df. Returns y_deseason, season_plot_ax, and the fitted DeterministicProcess instance.
    You might want to plot a seasonal_plot or plot_periodogram of y_deseason to check if the chosen parameters are reasonable.

    """
    
    if fourier is None:
        fourier = CalendarFourier(freq=season_freq, order=fourier_order)
    
    if dp is None:
        dp = DeterministicProcess(
        index=df.index,
        constant=True,
        order=dp_order,
        additional_terms=[fourier],
        drop=dp_drop,
        **DeterministicProcesskwargs
        )
    
    X = dp.in_sample()
    model = model.fit(X, df)
    y_pred = pd.Series(
                        model.predict(X),
                        index=X.index,
                        name=df.name+'_pred'
    )

    season_plot_ax = df.plot(**plot_params, alpha=0.5, title=df.name.replace('_',' '), ylabel=df.name)
    season_plot_ax = y_pred.plot(ax=season_plot_ax, label="Seasonal")
    season_plot_ax.legend();
    
    y_pred.name = df.name
    y_deseason = df - y_pred
    y_deseason.name = df.name +'_deseasoned'
    return y_deseason, y_pred, season_plot_ax, dp





# Making Features and target

def make_lags(df, n_lags=1, lead_time=1, out_3d=False):
    """
    Compute lags of a pandas.Series from lead_time to lead_time + n_lags. Alternatively, a list can be passed as n_lags.
    Returns a pd.DataFrame whose ith column is either the i+lead_time lag or the ith element of n_lags.
    """
    if isinstance(n_lags,int):
        lag_list = range(lead_time, n_lags+lead_time)
    else:
        lag_list = n_lags


    if out_3d or isinstance(df, pd.DataFrame):
        lags=list()
        for i in lag_list:
            df_lag = df.shift(i)
            if i!=0:                
                df_lag.columns = [f'{col}_lag_{i}' for col in df.columns]
            lags.append(df_lag) 
    else:
        lags ={
            f'{df.name}_lag_{i}': df.shift(i) for i in lag_list
            }

    
    return  pd.concat(lags,axis=1)





def make_multistep_target(ts:pd.Series, steps:int) -> pd.DataFrame:
    if steps==0:
        return ts
    
    if isinstance(steps,int):
        step_list = range(1,steps+1)
    else:
        step_list = steps
        
    step_dict = {f'{ts.name}_step_{i}': ts.shift(-i)
         for i in step_list}    
          
    return pd.concat(step_dict, axis=1)




def prepare_univariate_data(series, lags=1, steps=1, lead=1, test_size=0.2, return_Xy=False):
    """
    Create lags lag features and steps multi step targets. 
    If return_Xy is True, returns the resulting features 
    and targets, otherwise, return the train_test_split of them.
    """
    from sklearn.model_selection import train_test_split

    X = make_lags(series, n_lags=lags, lead_time=lead)
    y = make_multistep_target(series, steps)
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=test_size, shuffle=False)

    if return_Xy:
        return X, y
    
    return X_train, X_valid, y_train, y_valid





                        
def multi_wrapper(function, df: pd.DataFrame, 
columns_list:list =None, 
drop_original:bool =False, 
get_coordinate:int =None, 
output_3d:bool = False,
**kwargs)->pd.DataFrame:
    if columns_list is None:
        columns_list=df.columns
    
    X_list = list()

    for col in df.columns:
        if not drop_original:
            X_list.append(df[col])
        if col in columns_list:
            if get_coordinate is None:
                X_list.append(function(df[col], **kwargs))
            else:
                X_list.append(function(df[col], **kwargs)[get_coordinate])
    
    XX=pd.concat(X_list,axis=1)
    
    return XX


import numpy as np

def add_dim(df, timesteps:int =1)->np.array:
    """
    Transforms a pd.DataFrame into a 3D array with dimensions (samples, timesteps, faetures)
    """
    df = np.array(df)
    array_3d = df.reshape(df.shape[0],timesteps ,df.shape[1]//timesteps)
    return array_3d                        

def prepare_multivariate_data(df, y_col, lags=1, steps=1, lead=1, test_size=0.2, return_Xy=False, 
drop_columns:list =None, normalize:bool =False, output_3D:bool =False):
    """
    Create lags lag features and steps multi step targets. 
    If output_3D is True, return the data in a 3D array.
    If return_Xy is True, returns the resulting features 
    and targets, otherwise, return the train_test_split of them.
    """
    if isinstance(lags, int):
        lags = range(lead,lead+lags)


    if output_3D:
        df_1 = make_lags(df, n_lags=lags, lead_time=lead, out_3d=True)
    else:
        df_1 = multi_wrapper(make_lags, df, n_lags=lags, lead_time=lead)


    if steps==0:
        if drop_columns is None:
            drop_columns = df.columns.drop(y_col)
        X = df_1.drop(columns=drop_columns).dropna()
        y = make_multistep_target(X.pop(y_col),steps=steps).dropna()
    else:
        if drop_columns is None:
            drop_columns = []
        X = df_1.drop(columns=drop_columns).dropna()
        y = make_multistep_target(X[y_col],steps=steps).dropna()


    X, y = X.align(y, join='inner',axis=0)

    if return_Xy:
        return X, y
    
    from sklearn.model_selection import train_test_split    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=test_size, shuffle=False)

    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler().fit(X_train)
        X_train = mms.transform(X_train)
        X_valid = mms.transform(X_valid)



    if output_3D:
        X_train = add_dim(X_train, timesteps=len(lags))
        X_valid = add_dim(X_valid, timesteps=len(lags))

    return X_train, X_valid, y_train, y_valid
    
    

def prepare_3D_data(df, target_name, n_lags, n_steps, lead_time, test_size, normalize=True):
    '''
    Prepare data for LSTM - type of models.
    '''
        
    if isinstance(n_steps,int):
        n_steps = range(1,n_steps+1)
    
    n_steps = [-x for x in list(n_steps)]

    X = make_lags(df, n_lags=n_lags, lead_time=lead_time).dropna()
    y = make_lags(df[[target_name]], n_lags=n_steps).dropna()
    
    X, y = X.align(y, join='inner', axis=0)
    
    from sklearn.model_selection import train_test_split    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)

    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler().fit(X_train)
        X_train, X_val = mms.transform(X_train), mms.transform(X_val)    
    
    if isinstance(n_lags,int):
        timesteps = n_lags
    else:
        timesteps = len(n_lags)
    
    return add_dim(X_train,timesteps), add_dim(X_val,timesteps), y_train, y_val


def replace_columns(function, df, column_list, **functionkwargs):
    pass


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def apply_univariate_prediction(series,  test_size, to_predict=1, nlags=20, minimal_pacf=0.1, model=XGBRegressor(n_estimators=50)):
    '''
    Starting from series, breaks it in train and test subsets; 
    chooses which lags to use based on pacf > minimal_pacf; 
    finally it applies the given sklearn-type model. 
    Returns the resulting features and targets and the trained model. 
    It plots the graph of the training and prediction, together with their r2_score.
    '''
    s = series.iloc[:-test_size]
    
    if isinstance(to_predict,int):
        to_predict = [to_predict]
    
    from statsmodels.tsa.stattools import pacf

    s_pacf = pd.Series(pacf(s,nlags=nlags))
    
    column_list = s_pacf[s_pacf>minimal_pacf].index
    
    X = make_lags(series, n_lags=column_list).dropna()
    
    y = make_lags(series,n_lags=[-x for x in to_predict]).loc[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    model.fit(X_train,y_train)

    predictions_train = pd.DataFrame(
                        model.predict(X_train),
                        index=X_train.index,
                        columns=['Train Predictions']
    )


    predictions_test = pd.DataFrame(
                        model.predict(X_test),
                        index=X_test.index,
                        columns=['Test Predictions']
    )


    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5), sharey=True)

    y_train.plot(ax=ax1, legend=True)
    predictions_train.plot(ax=ax1)
    ax1.set_title('Train Predictions')

    y_test.plot(ax=ax2, legend=True)
    predictions_test.plot(ax=ax2)
    ax2.set_title('Test Predictions')
    plt.show()


    print(f'R2 train score: {r2_score(y_train[:-1],predictions_train[:-1])}')

    print(f'R2 test score: {r2_score(y_test[:-1],predictions_test[:-1])}')
    
    return X, y, model



# Plots

# Use pair plots to analyse correlation and stats.zscore for outliers

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None, q=.95):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    quantiles = np.floor(freqencies[(spectrum>np.quantile(spectrum, q=q))&(freqencies>1)])
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticks( quantiles, minor=True )

    ax.set_xticklabels( quantiles, rotation=90, minor=True, color='red' )

    
    ax.tick_params( axis='x', which='minor', direction='out', length=90, bottom='off' )
    
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=90,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

def compare_periodogram(y, y_deseason, figsize=(10,7)):
    """
    Plot side-by-side (well, not really) periodograms of y and y_deaseson, in order to verify deaseasonability.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
    ax1 = plot_periodogram(y, ax=ax1)
    ax1.set_title("Product Sales Frequency Components")
    ax2 = plot_periodogram(y_deseason, ax=ax2)
    ax2.set_title("Deseasonalized")
    return fig




def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, leads=False, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil((lags+int(leads)+bool(leads)) / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k < leads:
            j = leads-k
            ax = lagplot(x, y, lag=-j, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lead {j}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        if leads and k==leads:
            ax = lagplot(x, y, lag=0 , ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {0}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        if k >= leads+bool(leads):
            ax = lagplot(x, y, lag=k -leads +1-bool(leads), ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k -leads+1-bool(leads)}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig






# Hybrid Model


class BoostedHybrid:
    
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

        
    def fit(self, X_1, X_2, y):
        # YOUR CODE HERE: fit self.model_1
        self.model_1.fit(X_1,y)

        y_fit = pd.DataFrame(
            # YOUR CODE HERE: make predictions with self.model_1
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns,
        )

        # YOUR CODE HERE: compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        # YOUR CODE HERE: fit self.model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid
        return self

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            # YOUR CODE HERE: predict with self.model_1
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long

        # YOUR CODE HERE: add self.model_2 predictions to y_pred
        y_pred += self.model_2.predict(X_2)

        return y_pred.unstack()  # long to wide



def get_season(series: pd.Series, 
               test_size,
               season_freq='A', 
               fourier_order=0, 
               constant=True, 
               dp_order=1,  
               dp_drop=True,
               model1=LinearRegression(),
               fourier=None,
               is_seasonal=False,
               season_period=None,
               dp=None):
    """
    Decompose series in a deseasonalized and a seasonal part. The parameters are relative to the fourier and DeterministicProcess used. 
    Returns y_ds and y_s.

    """

    se = series.iloc[:-test_size]
    
    if fourier is None:
        fourier = CalendarFourier(freq=season_freq, order=fourier_order)

    if dp is None:
        dp = DeterministicProcess(
        index=se.index,
        constant=True,
        order=dp_order,
        additional_terms=[fourier],
        drop=dp_drop,
        seasonal=is_seasonal,
        period=season_period
        )

    X_in = dp.in_sample()
    X_out = dp.out_of_sample(test_size)
    
    model1 = model1.fit(X_in, se)
    
    X = pd.concat([X_in,X_out],axis=0)

    
    y_s = pd.Series(
                        model1.predict(X),
                        index=X.index,
                        name=series.name+'_pred'
    )

    y_s.name = series.name
    y_ds = series - y_s
    y_ds.name = series.name +'_deseasoned'
    return y_ds, y_s




def prepare_data(series, 
              test_size, 
              to_predict=1, 
              nlags=20,
              minimal_pacf=0.1):
    '''
    Creates a feature dataframe by making lags and a target series by a negative to_predict-shift. 
    Returns X, y.
    '''
    s = series.iloc[:-test_size]

    if isinstance(to_predict,int):
        to_predict = [to_predict]

    from statsmodels.tsa.stattools import pacf

    s_pacf = pd.Series(pacf(s,nlags=nlags))

    column_list = s_pacf[s_pacf>minimal_pacf].index

    X = make_lags(series, n_lags=column_list).dropna()

    y = make_lags(series,n_lags=[-x for x in to_predict]).loc[X.index].squeeze()


    return X, y
    
    
def get_hybrid_univariate_prediction(series: pd.Series, 
                                     test_size,
                                     season_freq='A', 
                                     fourier_order=0, 
                    constant=True, 
                                     dp_order=1,  
                                     dp_drop=True,
                    model1=LinearRegression(),
                    fourier=None,
                                     is_seasonal=False,
                                     season_period=None,
                                         dp=None, 
                                         to_predict=1, 
                                         nlags=20, 
                                         minimal_pacf=0.1, 
                                         model2=XGBRegressor(n_estimators=50)
                                         
                    ):
    """
    Apply the hybrid model method by deseasonalizeing/detrending a time series with model1 and investigating the resulting series with model2. It plots the respective graphs and compute r2_scores. 
    """
    
    
    
    y_ds, y_s = get_season(series, test_size, season_freq=season_freq, fourier_order=fourier_order, constant=constant, dp_order=dp_order, dp_drop=dp_drop, model1=model1, fourier=fourier, dp=dp, is_seasonal=is_seasonal, season_period=season_period)
    
    X, y_ds = prepare_data(y_ds,test_size=test_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_ds, test_size=test_size, shuffle=False)
    
    y = y_s.squeeze() + y_ds.squeeze()
        
    model2 = model2.fit(X_train,y_train)

    predictions_train = pd.Series(
                        model2.predict(X_train),
                        index=X_train.index,
                        name='Prediction'
    )+y_s[X_train.index]


    predictions_test = pd.Series(
                        model2.predict(X_test),
                        index=X_test.index,
                        name='Prediction'
    )+y_s[X_test.index]

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5), sharey=True)

    y_train_ps = y.loc[y_train.index]
    y_test_ps = y.loc[y_test.index]

    y_train_ps.plot(ax=ax1, legend=True)
    predictions_train.plot(ax=ax1)
    ax1.set_title('Train Predictions')

    y_test_ps.plot(ax=ax2, legend=True)
    predictions_test.plot(ax=ax2)
    ax2.set_title('Test Predictions')
    plt.show()


    print(f'R2 train score: {r2_score(y_train_ps[:-to_predict],predictions_train[:-to_predict])}')

    print(f'R2 test score: {r2_score(y_test_ps[:-to_predict],predictions_test[:-to_predict])}')




# Multioutput model

from sklearn.multioutput import MultiOutputRegressor  # This is a wrapper for any regression model with multiple outputs (i.e., y has more then one column). It creates an independent Regressor for each column.


class MultiModelRegressor:
    def __init__(self, models: list =[LinearRegression()]) -> None:
        self.models=models  
        self.y_columns = None
        self.n_models = len(models)
        
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        if self.n_models!=len(y.columns):
            raise ValueError(f'Number of models is expected to be the same number of y.columns, instead they are {self.n_models} and {len(y.columns)}')

        self.models = [self.models[i].fit(X,y.iloc[:,i])  for i in range(self.n_models)]
        self.y_columns = y.columns
        return self

    def predict(self, X):
        y_pred = pd.DataFrame(
            {self.y_columns[i]:self.models[i].predict(X) for i in range(self.n_models)},
            index=X.index
        )
        return y_pred


#  Recursive Strategy Should I implement this thing? (Commented code below not implemented)


# class RecursiveWarper:
#     def __init__(self, model, n_lags=1, n_steps=1, lead_time=1) -> None:
#         self.model=model  
#         self.y_columns = None
#         self.n_lags = n_lags
#         self.n_steps = n_steps
#         self.lead_time = lead_time
        
        
        
#     def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
#         self.model.fit(X,y)
#         self.y_columns = y.columns
#         return self
   
#     def predict_nsteps(self, X, n_steps:int =1):
        

# 
# def recursive_training(model, X, y, lags=1, steps=1, lead=1):
#     dp = DeterministicProcess(
#         index=X.index,
#         constant=True,
#         order=1
#     )
#     time_delta = X.index[-1]-X.index[-2]
#     X = make_lags(X,lags=lags, lead_time=lead)
#     y = make_multistep_target(y, steps=steps)
#     model.fit(X)

#     pd.DataFrame(model.predict(dp.out_of_sample(1)))





# DirRec Method. Original article: https://research.cs.aalto.fi//aml/Publications/Publication64.pdf

## For single-model DirRec Method, wrap it in:
# from sklearn.multioutput import RegressorChain
## If one needs multi-model, use the warpper below.

class MultiRegressorChain:
    def __init__(self, estimators, *, order=None, cv=None, random_state=None):
        self.estimators = estimators
        self.order = order
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Y : array-like of shape (n_samples, n_classes)
            The target values.
        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")
        # Below is the only line I changed in sklearn _BaseChain class, apart from the base_estimator
        self.estimators_ = self.estimators

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(X_aug[:, : (X.shape[1] + chain_idx)], y, **fit_params)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx], y=y, cv=self.cv
                )
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred
    
    
    
# Vanilla LSTM

