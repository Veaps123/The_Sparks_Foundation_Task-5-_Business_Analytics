#!/usr/bin/env python
# coding: utf-8

# In[53]:


#import library for packages
import pandas as pd
import numpy as np

#import library for visualiztion 
import matplotlib.pyplot as plt
import seaborn as sns

# import Data from csv file
data = pd.read_csv("C:\\Users\\hp\\Downloads\\SampleSuperstore.csv")
data


# In[54]:


# taking Top Fifteen rows*colunms
data.head(15)


# In[55]:


#data information 
data.info()


# In[56]:


# mathematical standard description (Dataset Statistic Figures All Columns)
data.describe().all


# In[57]:


# Changing Incurrect Data Type to object
data['Postal Code'] = data['Postal Code'].astype('object')


# In[58]:


# Changing Incurrect Data Type to object
data['Quantity'] = data['Quantity'].astype('float64')


# In[59]:


#find the duplicated data
data.duplicated().sum()


# In[60]:


data.isnull().sum()


# In[61]:


# To drop duplicate data
data.drop_duplicates(subset = None, keep = 'first' , inplace =True)
data


# In[62]:


# to get correlation
correlation = data.corr()
correlation


# In[63]:


#Sales Statistic Figures

sales_count = data["Sales"].count()
sales_mean = data["Sales"].mean()
Sales_Standard_Deviation = data["Sales"].std()
Sales_Min = data["Sales"].min()
print(sales_count)
print(sales_mean)
print(Sales_Standard_Deviation)
print(Sales_Min)


# In[64]:


# Let's create a class

def display_all(data):
    
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(data)


# In[ ]:





# In[65]:


sns.heatmap(correlation,annot= True)


# In[66]:


data.hist(bins=10,figsize=(30,15))
plt.show()


# In[67]:


sns.countplot(data['Ship Mode'])


# In[68]:


sns.countplot(data['Segment'])


# In[69]:


sns.countplot(data['Country'])


# In[70]:


sns.countplot(data['Region'])


# In[71]:


sns.countplot(data['Sub-Category'])


# In[72]:


sns.countplot(data['Category'])


# In[73]:


sns.countplot(data['State'])


# In[74]:


plt.figure(figsize=(15,10))
data['Sub-Category'].value_counts().plot(kind='pie', autopct = '%1.1f%%')
plt.show()


# In[75]:


sns.countplot( x = data['State'], order = (data['State'].value_counts().head(20)).index )
plt.xticks(rotation =90)


# In[76]:


states = data.groupby('State')[['Sales','Profit']].sum().sort_values(by='Sales',ascending=False)
plt.figure(figsize=(40,40))


# In[77]:


states = data.groupby("State")[['Sales','Profit']].sum().sort_values(by = 'Sales', ascending = False)
plt.figure(figsize = (25,20))
states[:25].plot(kind = 'bar' , edgecolor = '#000000',color=['red','yellow'])
plt.title('Profit/Loss & Sales of top 25 States')
plt.xlabel('States')
plt.ylabel('Sales and Net Profit/Loss')
plt.grid(True)

states[25:].plot(kind = 'bar' , edgecolor = '#000000', color = ['red','yellow'])
plt.title('Profit/Loss & Sales of below 25 States')
plt.xlabel('States')
plt.ylabel('Sales and Net Profit/Loss')
plt.grid(True)


# In[78]:


sub_cat = data.groupby('Sub-Category')[['Sales','Profit']].sum().sort_values(by = 'Sales', ascending = False)
sub_cat.plot(kind = 'bar' , color = ['blue','orange'], edgecolor = '#000000')


# pd.DataFrame()

# In[79]:


pd.DataFrame(data.groupby('State').sum())['Profit'].sort_values(ascending = True)


# In[80]:


# MODEL 
pd.DataFrame(data.groupby('State').sum())['Discount'].sort_values(ascending = True)


# In[81]:


furniture = data.loc[data['Category'] == 'Furniture']


# In[82]:


furniture['Profit'].min(), furniture['Profit'].max()


# In[83]:


furniture.head(5)


# In[84]:


cols = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Category', 'Sub-Category', 'Quantity', 'Discount']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Profit')
furniture.isnull().sum()


# In[85]:


furniture


# In[86]:


furniture = furniture.groupby('Profit')['Sales'].sum().reset_index()


# In[87]:


furniture.shape


# In[88]:


furniture.info()


# In[89]:


furniture = furniture.set_index('Profit')
furniture.index


# In[90]:


y = furniture['Sales']


# In[91]:


y.plot(figsize=(10, 4))
plt.show()


# In[92]:


#FIT THE MODEL
from pylab import rcParams
import statsmodels.api as sm


# In[93]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[94]:


results.plot_diagnostics(figsize = (12,6))
plt.show()


# In[95]:


#Thank You


# In[ ]:




