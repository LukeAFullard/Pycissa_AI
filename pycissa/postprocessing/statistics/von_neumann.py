
import numpy as np
import copy
import warnings

def von_neumann_test(x:        np.ndarray,
                     alpha:    float = 0.05,
                     version:  str = "two sided",
                     unbiased: bool =True):
    '''
    The von Neumann Successive Difference Test: nonparametric test for randomness.
    Tests for the presence of lag 1 autocorrelation.
    Null hypothesis that the time series data is independent and normally
    distributed.
    This Python implementation is loosly based off an R implementation in the DescTools package:
    https://andrisignorell.github.io/DescTools/reference/VonNeumannTest.html

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input array for testing
    alpha : float, optional
        DESCRIPTION. Confidence level. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
    version : str, optional
        DESCRIPTION. Type of test to perform. Options are "two sided", "one sided lower", "one sided upper"
                        The default is "two sided".
    unbiased : bool, optional
        DESCRIPTION. Whether to use the unbiased version of the test The default is True.

    Returns
    -------
    von_neumann_ratio : float
        DESCRIPTION: The value of the von-Neumann test statistic
    p_value : float
        DESCRIPTION: The p-value of the test statistic
    reject_null_hypothesis : bool
        DESCRIPTION: Did we reject the null hypothesis (True) or not (False)
    intepretation : str
        DESCRIPTION: Text discussing the test result

    '''
    #Von Neumann Successive Difference Test: nonparametric test for randomness
    import numpy as np
    import scipy
    import copy
    n = len(x)
    x_copy = copy.deepcopy(x)
    x_copy = np.array(x_copy)
    x_copy = x_copy.reshape(n,)

    differencing = np.diff(x_copy)
    x_mean = np.mean(x_copy)

    if unbiased:
        von_neumann_ratio = sum(np.power(differencing,2.)) / sum(np.power(x - x_mean,2.)) * n/(n-1.)
        expectation_value = 2. * n/(n-1.)
        variance          = 4. * np.power(n,2.) * (n-2.) / ((n+1.) * np.power(n-1.,3.))

    else:
        von_neumann_ratio = sum(np.power(differencing,2.)) / sum(np.power(x - x_mean,2.))
        expectation_value = 2.
        variance          = 4. * (n - 2.) / ((n + 1.)*(n - 1.))


    z_statistic       = (von_neumann_ratio - expectation_value) / np.sqrt(variance)

    #find p-value
    p_value = 1.
    if version == 'two sided':
        p_value = scipy.stats.norm.sf(abs(z_statistic))*2
    if version == 'one sided lower':
        p_value = 1- scipy.stats.norm.sf(z_statistic)
    if version == 'one sided upper':
        p_value = scipy.stats.norm.sf(z_statistic)

    reject_null_hypothesis = p_value < alpha
    intepretation = f'''The {version} von Neumann ratio test was perfomed with the null hypothesis that the time series data is independent and normally
distributed. The null hypothesis was not able to be rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data may be independent (i.e. is likely not autocorrelated).'''
    if reject_null_hypothesis:
        intepretation = f'''The {version} von Neumann ratio test was perfomed with the null hypothesis that the time series data is independent and normally
distributed. The null hypothesis was rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data is not independent (i.e. is autocorrelated).'''


    return von_neumann_ratio, p_value, reject_null_hypothesis,intepretation

###############################################################################
###############################################################################
###############################################################################
def rank_von_neumann_test(x:        np.ndarray,
                     alpha:    float = 0.05,
                     version:  str = "two sided",
                     pvalue:   str = 'auto'):
    '''
    The Rank von Neumann Successive Difference Test: nonparametric test for randomness.
    Tests for the presence of lag 1 autocorrelation.
    Null hypothesis that the time series data is independent and normally
    distributed.
    This Python implementation is loosly based off an R implementation in the DescTools package:
    https://github.com/cran/DescTools/blob/master/R/Tests.r

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input array for testing
    alpha : float, optional
        DESCRIPTION. Confidence level. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
    version : str, optional
        DESCRIPTION. Type of test to perform. Options are "two sided", "one sided lower", "one sided upper"
                        The default is "two sided".
    pvalue : str, optional
        DESCRIPTION. How to calculate the p value. Options are "auto", "beta", "Normal". The default is 'auto'.

    Returns
    -------
    rank_von_neumann : float
        DESCRIPTION: The value of the Rank von-Neumann test statistic
    p_value : float
        DESCRIPTION: The p-value of the test statistic
    reject_null_hypothesis : bool
        DESCRIPTION: Did we reject the null hypothesis (True) or not (False)
    intepretation : str
        DESCRIPTION: Text discussing the test result

    '''
    x_rank_ = copy.deepcopy(x)
    x_rank_ = np.array(x_rank_)

    #sort the data into rank form
    x_rank_ = np.sort(x_rank_)
    x_rank = [list(x_rank_).index(a) +1 for a in x]

    n = len(x_rank)

    #check percentage of the observations that are ties
    percent_ties = 100*(n - len(np.unique(x_rank)))/n
    if percent_ties > 0: warnings.warn(f"{percent_ties}% of observations are ties. The p-value may not be accurate")

    #check enough data present
    if (n < 10):
        warnings.warn(f"Sample size must be greater than 9. Only {n} observations found. Rank von-Neumann statistic not calculated.")
        return (None,)*4

    rank_von_neumann = (sum(np.power(np.diff(x_rank),2))) / (sum(np.power(x_rank,2)) - n * (np.power(np.mean(x_rank),2)))

    rvn_mean = 2.
    rvn_variance = (4*(n-2)*((5*np.power(n,2))-(2*n)-9))/(5*n*(n+1)*np.power(n-1,2))

    #calculate p-value
    if pvalue == "auto":
      pvalue = "normal"
      if n<= 100:
          pvalue = "beta"

    if pvalue == "beta":
        from scipy.stats import beta
        beta_parameter = (5*n*(n+1)*np.power(n-1,2))/(2*(n-2)*((5*np.power(n,2))-(2*n)-9))-1/2
        pv0 = beta(a=beta_parameter, b=beta_parameter).cdf(rank_von_neumann/4)

    if pvalue=="normal":
        from scipy.stats import norm
        pv0 = norm.cdf((rank_von_neumann - rvn_mean)/np.sqrt(rvn_variance))

    if (version=="two sided"):
        p_value = 2 * min(pv0, 1 - pv0)
    if (version=="one sided lower"):
        p_value = 1 - pv0
    if (version=="one sided upper"):
        p_value =  pv0

    reject_null_hypothesis = p_value < alpha
    intepretation = f'''The {version} Rank von Neumann ratio test was perfomed with the null hypothesis that the time series data is independent and normally
distributed. The null hypothesis was not able to be rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data may be independent (i.e. is likely not autocorrelated).'''
    if reject_null_hypothesis:
        intepretation = f'''The {version} Rank von Neumann ratio test was perfomed with the null hypothesis that the time series data is independent and normally
distributed. The null hypothesis was rejected at the alpha = {alpha} level; the p value was {p_value}. Hence the data is not independent (i.e. is autocorrelated).'''


    # need to test this now and test p value vs alpha value
    return rank_von_neumann, p_value, reject_null_hypothesis,intepretation
