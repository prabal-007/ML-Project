#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


##### Importing the data

# In[2]:

df = pd.read_csv(r'C:\Users\admin\Downloads\Horse colic project\horse.csv')

df.head()


# ### Data Examination and Cleaning
# 

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


plt.figure(figsize = [10, 10])
nan_per  = df.isna().sum()/len(df)*100
plt.bar(range(len(nan_per)), nan_per)
plt.xlabel('Features')
plt.ylabel('% of NAN values')
plt.plot([0, 25], [40, 40] ,'r--', lw = 1)
plt.xticks(list(range(len(df.columns))), list(df.columns.values), rotation = 'vertical')


# ### Saperating Categorical and Numerical Data

# In[6]:


obj_columns = []
non_obj_columns = []
for column in df.columns.values:
    if df[column].dtype == 'object':
        obj_columns.append(column)
    else:
        non_obj_columns.append(column)
        
print(f'Object columns : {len(obj_columns)},\n\n{list(obj_columns)}; \n\nNon object columns : {len(non_obj_columns)},\n \n{list(non_obj_columns)}')

df_obj = df[obj_columns]
df_non_obj = df[non_obj_columns]


# ### Removing and Filling Missing Values in Numerical and Categorical Data

# In[7]:


# For columns with more than 40% NAN Value : Remove Columns
# For columns with less than 40% NAN Value :
#     For Numerical Data: Replace NAN values with median value of that particular column
#     For Categorical Data: Replace NAN values with mode value of that particular column

print(f"Data Size Before Numerical NAN Column(>40%) Removal : {df_non_obj.shape}")
for col in df_non_obj.columns.values:
    if (pd.isna(df_non_obj[col]).sum())>0:
        if pd.isna(df_non_obj[col]).sum() > (40/100*len(df_non_obj)):
            print(col,"removed")
            df_non_obj = df_non_obj.drop([col], axis=1)
        else:
            df_non_obj[col]=df_non_obj[col].fillna(df_non_obj[col].median())
f'Data Size After Numerical NAN Column(>40%) Removal : {df_non_obj.shape}'


# #### Converting Categorical Data to Numerical and Merging Them

# In[8]:


for col in df_obj.columns.values:
    df_obj[col]=df_obj[col].astype('category').cat.codes
df_merge=pd.concat([df_non_obj,df_obj],axis=1)

target=df['outcome']
print(target.value_counts())
target=df_merge['outcome']
print(target.value_counts())


# #### Correlation between Features and Outcome

# In[9]:


import seaborn as sns

plt.figure(figsize = [10,8])
train_corr=df_merge.corr()
sns.heatmap(train_corr, vmax=0.8)
corr_values=train_corr['outcome'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
print(f"Correlation of mentioned features wrt outcome in ascending order : \n {abs(corr_values).sort_values(ascending=False)}")


# In[10]:


# Removing unwanted very less correlated features

print("Data Size Before Correlated Column Removal :",df_merge.shape)

for col in range(len(corr_values)):
        if abs(corr_values[col]) < 0.1:
            df_merge=df_merge.drop([corr_values.index[col]], axis=1)
            print(corr_values.index[col],"removed")
            
print("Data Size After Correlated Column Removal :",df_merge.shape)


# In[11]:


#packed_cell_volume

col='packed_cell_volume'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])

y=df_merge[col][target==2]
x=df_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=df_merge[col][target==0]
x=df_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=df_merge[col][target==1]
x=df_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()


# In[12]:


#pulse 
col='pulse'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])
y=df_merge[col][target==2]
x=df_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=df_merge[col][target==0]
x=df_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=df_merge[col][target==1]
x=df_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()


# In[13]:


df_merge.head()


# #### Data Standardization

# In[14]:


df.describe()


# In[15]:


from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(df_merge)


# #### Covariance Matrix

# In[16]:


print('Covariance matrix: \n', np.cov(x_std.T))


# #### Eigen Decomposition of Covariance Matrix

# In[17]:


cov_mat = np.cov(x_std.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
print('Eigen vectors \n',eigen_vectors)
print('\nEigen values \n',eigen_values)


# In[18]:


pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs.sort()
pairs.reverse()

print('Eigen Values in descending order:')
for i in pairs:
    print(i[0])


# In[19]:


tot = sum(eigen_values)
var_per = [(i / tot)*100 for i in sorted(eigen_values, reverse=True)]
cum_var_per = np.cumsum(var_per)

plt.figure(figsize=(10,10))
x=['PC %s' %i for i in range(1,len(var_per))]
ind = np.arange(len(var_per)) 
plt.bar(ind,var_per)
plt.xticks(ind,x);
plt.plot(ind,cum_var_per,marker="o",color='orange')
plt.xticks(ind,x);


# #### Projection onto New Feature Space¶

# In[20]:


N=16
value=10
a = np.ndarray(shape = (N, 0))
for x in range(1,value):
    b=pairs[x][1].reshape(16,1)
    a = np.hstack((a,b))
print("Projection Matrix:\n",a)


# In[21]:


Y = x_std.dot(a)


# In[22]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for name in ('died', 'euthanized', 'lived'):
    plt.scatter(x=Y[name,3], y=Y[name,4])
plt.legend( ('died', 'euthanized', 'lived'))
plt.title('After PCA')

plt.subplot(1,2,2)
for name in ('died', 'euthanized', 'lived'):
    plt.scatter(
        x=Xstd[temp==name,3],
        y=Xstd[temp==name,4]
    )
plt.title('Before PCA')
plt.legend( ('died', 'euthanized', 'lived'))


# In[ ]:
