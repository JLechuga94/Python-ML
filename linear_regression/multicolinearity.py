import pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Calculation of VIF to determine colinearity between prediction variables

data = pandas.read_csv("../../datasets/ads/Advertising.csv")

# Newspaper - TV + Radio --- R^2    VIF = 1/(1-R^2)
lm_news = smf.ols(formula='Newspaper ~ TV + Radio', data = data).fit()
rsquared_n = lm_news.rsquared
VIF = 1/(1-rsquared_n)
print(VIF)

# TV - Newspaper + Radio --- R^2    VIF = 1/(1-R^2)
lm_tv = smf.ols(formula='TV ~ Newspaper + Radio', data = data).fit()
rsquared_n = lm_tv.rsquared
VIF = 1/(1-rsquared_n)
print(VIF)

# Radio - TV + Newspaper --- R^2    VIF = 1/(1-R^2)
lm_radio = smf.ols(formula='Radio ~ TV + Newspaper', data = data).fit()
rsquared_n = lm_radio.rsquared
VIF = 1/(1-rsquared_n)
print(VIF)
