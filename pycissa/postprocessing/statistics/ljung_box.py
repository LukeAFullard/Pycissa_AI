import numpy as np
import statsmodels.api as sm


def run_ljung_box_test(x:         np.ndarray,
                       lags:      int = 1,
                       return_df: bool = False,
                       alpha:     float = 0.05):
    '''
    Ljung box test for hypothesis test for autocorrelation in a time series.
    Null hypothesis is that the data are independently distributed (i.e. have no auto correlation).

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input array for testing
    lags : int|list, optional
        DESCRIPTION. Number of lags to test. If lags is an integer then all integer lags less than this number are also tested. e.g. lags = 3 will test lags 1, 2, and 3.
                        If lags is a list then only the integers in the list are tested. e.g. lags = [3] will test only lags 3.
                        The default is 1.
    return_df : bool, optional
        DESCRIPTION. Return a dataframe or not (probably not necessary). The default is False.
    alpha : float, optional
        DESCRIPTION. Confidence level. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).

    Returns
    -------
    ljung_box_result : dict
        DESCRIPTION: dictionary of results at each lag
    rejected_pvalues : list
        DESCRIPTION: list of pvalues where the null hypothesis was rejected. Order corresponds to that of rejected_lags
    rejected_lags : list
        DESCRIPTION: lags at which the null hypothesis was rejected (i.e. lags which showed evidence of autocorrelation)
    reject_any_null_hypothesis : bool
        DESCRIPTION: Did we reject the null hypothesis (True) or not (False)
    intepretation : str
        DESCRIPTION: Text discussing the test result

    '''

    #annoyingly, this is returned as a Pandas dataframe, not as numpy arrays
    ljung_box_r = sm.stats.acorr_ljungbox(x, lags=lags, return_df=return_df)
    ljung_box_result = {}

    reject_any_null_hypothesis = False
    rejected_lags    = []
    rejected_pvalues = []

    #iterate through the results for each lag
    for index,stat,pval in zip(ljung_box_r.index,ljung_box_r.lb_stat,ljung_box_r.lb_pvalue):
        #check if the null is rejected for this lag
        reject_null_hypothesis = pval < alpha
        ljung_box_result.update({index:{
                                    'ljung_box_statistic'   :stat,
                                    'p_value'               :pval,
                                    'reject_null_hypothesis':reject_null_hypothesis,
                                    }})
        #if the null is rejected, make a note and at which lag
        if reject_null_hypothesis:
            reject_any_null_hypothesis = True
            rejected_lags.append(index)
            rejected_pvalues.append(pval)

    intepretation = f'''The Ljung-Box test was perfomed with the the null hypothesis is that the data are independently distributed (i.e. have no auto correlation).
    The null hypothesis was not able to be rejected at the alpha = {alpha} level. Hence the data show no evidence of autocorrelation.'''

    if reject_any_null_hypothesis:
        intepretation = f'''The Ljung-Box test was perfomed with the the null hypothesis is that the data are independently distributed (i.e. have no auto correlation).
        The null hypothesis was rejected at the alpha = {alpha} level at lags {rejected_lags}; the p values which implied rejection were {rejected_pvalues}. Hence the data show evidence of autocorrelation.'''

    return ljung_box_result, rejected_pvalues, rejected_lags,  reject_any_null_hypothesis, intepretation


