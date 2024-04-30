from scipy import stats
import numpy as np

def cal_std_err(x):
    num_macro_rep = len(x)
    
    unbiased_std_dev = (np.sum((x - np.mean(x))**2))/(num_macro_rep-1)
    std_err = unbiased_std_dev / num_macro_rep
    return std_err

def conf_interval(x, conf_at):
    """Calculate Confidence interval

    Args:
        x ([type]): [description]
        conf_at ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean, std  = x.mean(), x.std(ddof = 1)
    std_err = np.sqrt(cal_std_err(x)/len(x))
    conf_intveral = (mean - (stats.norm.ppf(1-(1-conf_at)/2)*std_err), mean+(stats.norm.ppf(1-(1-conf_at)/2)*std_err))

    
    # conf_intveral = stats.norm.interval(conf_at, loc=mean, scale=std)
    # print("***************************")
    # print(mean)
    # print(std)
    # print(std_err)
    # print(stats.norm.ppf(1-(1-conf_at)/2))
    # print(ci)
    # print(conf_intveral)
    # print("***************************")
    # print(grae)
    return conf_intveral