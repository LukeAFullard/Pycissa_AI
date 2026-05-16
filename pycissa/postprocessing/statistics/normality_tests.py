import numpy as np
from scipy import stats

def run_normality_test(x:          np.ndarray,
                       axis:       int = 0,
                       nan_policy: str = 'propagate',
                       alpha:      float = 0.05):
    '''
    Wrapper for scipy stats omnibus test of normality.

    Test whether a sample differs from a normal distribution.
    This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis to produce an omnibus test of normality.
    References
    [1](1,2)
    D’Agostino, R. B. (1971), “An omnibus test of normality for moderate and large sample size”, Biometrika, 58, 341-348
    [2](1,2)
    D’Agostino, R. and Pearson, E. S. (1973), “Tests for departure from normality”, Biometrika, 60, 613-622
    [3]
    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3/4), 591-611.
    [4]
    B. Phipson and G. K. Smyth. “Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn.” Statistical Applications in Genetics and Molecular Biology 9.1 (2010).
    [5]
    Panagiotakos, D. B. (2008). The value of p-value in biomedical research. The open cardiovascular medicine journal, 2, 97.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: The array containing the sample to be tested.
    axis : int, optional
        DESCRIPTION: Axis along which to compute test. Default is 0. If None, compute over the whole array x. The default is 0.
    nan_policy : str, optional
        DESCRIPTION: Defines how to handle when input contains nan. The following options are available (default is ‘propagate’):
                    ‘propagate’: returns nan
                    ‘raise’: throws an error
                    ‘omit’: performs the calculations ignoring nan values. The default is 'propagate'.
    alpha : float, optional
        DESCRIPTION: Confidence level. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).

    Returns
    -------
    test_statistic : float
        DESCRIPTION: s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
    p_value : float
        DESCRIPTION: A 2-sided chi squared probability for the hypothesis test.
    reject_null_hypothesis : bool
        DESCRIPTION: Did we reject the null hypothesis (True) or not (False)
    intepretation : str
        DESCRIPTION: Text discussing the test result

    '''
    test_statistic,p_value =  stats.normaltest(x,
                                          axis = axis,
                                          nan_policy = nan_policy)

    reject_null_hypothesis = p_value < alpha
    intepretation = f'''The normality test was perfomed with the null hypothesis that the sample comes from a normal distribution. The null hypothesis was not able to be rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data may be normal.'''
    if reject_null_hypothesis:
        intepretation = f'''The normality test was perfomed with the null hypothesis that the sample comes from a normal distribution. The null hypothesis was rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data is not normal'''


    return test_statistic,p_value, reject_null_hypothesis,intepretation
