#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("data/finSector.csv")


# In[4]:


df.rename(columns={"Unnamed: 0":"Date"},inplace=True)


# In[8]:


df['Date'] = pd.to_datetime(df['Date'])


# In[9]:


df.dtypes


# In[10]:


df['Date']<'2023-06-01'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


dfhead = df.set_index("Date")
df['MA_adjClose'].plot()
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Mastercard Price data")
plt.show()


# In[14]:


fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.plot(dfhead['MA_adjClose'])
ax1.set_title("Mastercard")
ax2.plot(dfhead['GS_adjClose'])
ax2.set_title("Goldmansacks")
ax3.plot(dfhead['AXP_adjClose'])
ax3.set_title("AXP")
ax4.plot(dfhead['MS_adjClose'])
ax4.set_title("MS")
ax5.plot(dfhead['V_adjClose'])
ax5.set_title("Visa")
plt.tight_layout()
plt.show()


# In[ ]:





# ## Returns (daily;monthly;weekly)

# In[ ]:





# In[47]:


MA_daily_returns = df['MA_adjClose'].pct_change()
GS_daily_returns = df['GS_adjClose'].pct_change()
AXP_daily_returns = df['AXP_adjClose'].pct_change()
MS_daily_returns = df['MS_adjClose'].pct_change()
V_daily_returns = df['V_adjClose'].pct_change()

## incorporate daily in data
df['MA_daily']=0 
df['MA_daily']= df['MA_adjClose'].pct_change()

df['GS_daily']=0 
df['GS_daily']= df['GS_adjClose'].pct_change()


# In[16]:


MA_monthly_returns = dfhead['MA_adjClose'].resample('M').ffill().pct_change()
GS_monthly_returns = dfhead['GS_adjClose'].resample('M').ffill().pct_change()
AXP_monthly_returns = dfhead['AXP_adjClose'].resample('M').ffill().pct_change()
MS_monthly_returns = dfhead['MS_adjClose'].resample('M').ffill().pct_change()
V_monthly_returns = dfhead['V_adjClose'].resample('M').ffill().pct_change()


# In[17]:


def logRet(MA_adjClose):
    return np.log(MA_adjClose).diff()
df['log ret_MA'] = df['MA_adjClose'].transform(logRet)


# In[48]:


df


# ### weekly return - mondays

# In[18]:


dfx=pd.read_csv("data/finSector.csv")


# In[38]:


dfx
dfx.rename(columns={"Unnamed: 0":"Date"},inplace=True)
dfx['Date'] = pd.to_datetime(dfx['Date'])
def rename(e):
    return e[0]+e[-1]
dfx['returns']=dfx['MA_adjClose']

dfxhead = dfx.set_index("Date")


# In[20]:


dfx["weekday"]=dfx['Date'].dt.weekday


# In[39]:


dfx=dfx.loc[dfx['weekday']<1]


# In[22]:


MA_daily_returns = dfx['MA_adjClose'].pct_change()


# In[40]:


def logRet(MA_adjClose):
    return np.log(MA_adjClose).diff()
dfx['log ret_MA'] = dfx['MA_adjClose'].transform(logRet)


# ## Histograms

# In[30]:


sns.barplot(x="Date", y="log ret_MA",
                data=dfx)


# In[ ]:


import statsmodels.api as sm


# In[119]:


ax = dfx.hist(column='log ret_MA', bins=100, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)


# In[ ]:


df1=df.resample('')


# test : 
# print(MA_daily_returns.head())
# print(MA_monthly_returns.head())

# In[124]:


df["MA_adjClose"].mean()
df["GS_adjClose"].mean()
df["AXP_adjClose"].mean()
df["MS_adjClose"].mean()
df["V_adjClose"].mean()


# ## Variance 

# In[139]:


MA = dfhead['MA_adjClose'].pct_change().apply(lambda x: np.log(1+x))
mean_MA = MA.sum()/MA.count()
sqd_MA = MA.apply(lambda x: (x-mean_MA)**2)
ssqd_MA = sqd_MA.sum()
var_MA = ssqd_MA/(MA.count()-1)
MA.var()


# ### White noise process

# In[4]:


np.random.normal(loc=0.0, scale=1.0, size=100)


# In[ ]:


list=[]
new=[]
for x in white_noise:
    item=0
    new=df.MA_adjClose[str(item)]+white_noise[str(item)]
    list.append(new[0])
    new.remove[0]
    item=item+1
print(list)

