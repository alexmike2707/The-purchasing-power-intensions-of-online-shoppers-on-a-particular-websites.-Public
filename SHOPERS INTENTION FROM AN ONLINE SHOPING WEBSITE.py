#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The dataset contains information of online shoppers pruchasing power intension,
# 
# The essence is to allow each session to belong to different user in a year period.
# 
# The primary purpose of the dataset is to predicts the purchasing intention of any visitor to this particular store/website

# # Description of varaibles on the rows;
# 
# Administrative ** NUmbers of pages vissited by a user
# 
# Administrative Duration ** Amount of time spent on a category of page
# 
# Informational ** it is the numbers of informational pages a user vissted
# 
# Informational Duration ** it is the amount of time spent in this category of page
# 
# ProductRelated ** it is the number of productrelated pages a user vissited
# 
# Productrelated Duration **it is the amount of time spent in this category of pages
# 
# BounceRates ** the percentage of persons who enters into  that sites throgh a page without trigering an additional task
# 
# ExitRates ** The percentage of pageviews on the website that end at that specific page.
# 
# PageValues ** pages where e_commerce transaction successfully completed
# 
# SpecialDay ** The closness of the browsing date to special day or holidays e.g(mothers day, Valentine day); it is asumed that transactions are more likely to be completed within this periods.
# 
# Month ** The month in which the page view occured
# 
# OperatingSystem ** The operating system the user was on when viewing the pages.
# 
# Browser ** The browser the user was using to view the pages
# 
# Region ** |The location of the user
# 
# Traffic Type ** The categories of traffic the user belong to
# 
# Vissitor Type ** the type of user, either vissitor, returning or others
# 
# Weekend ** days within the week, either working days or weekend
# 
# Revenue ** Assesment of whether the user complete the purchase or not
# 
# 

# In[1]:


# import neccessary libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading of dataset


data = pd.read_csv(r'C:\Users\Sirmiky\Downloads\Online Shoppers Intention\online_shoppers_intention.csv')
data.head(10)


# In[3]:


data.tail(10)


# In[4]:


# view the size of the dataset

data.shape


# In[5]:


#check for missimg values

data.isnull().sum()


# In[6]:


# get the statistics of the data

data.describe().T


# In[7]:


# summary statistics for categorigal column

data.describe(include='object').T


# In[8]:


# check for the data info

data.info()


# In[9]:


#Visualize Correlations in a heatmap

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True)


# ### Summary findings from the data set.before EDA's
# 
# * There are 12330 records and 18 features in the data set
# * There are no missing values in the data set
# * There are ten unique months within which potential buyers were viewing  the pages of the websites, and may has the highest freency
# * There are three categories of visitors to the sites. with returning visitors the highest frequency
# * There are about 705 maximum of product related pages in the data set.
# * E_commerce transactions were successfully completed in 362 pages

# # EDA

# ### Visitor type analysis

# In[10]:


# value count of the type of user & vissualization


data.VisitorType.value_counts()


# In[11]:


plt.figure(figsize=(8,6))
plt. title('Types of viewers/categories of viewers', fontsize=18,fontweight='bold') 
ax = sns.countplot(x = 'VisitorType', data = data)

for i in ax.containers:
    ax.bar_label(i)
plt.xlabel('categories of visitors', fontsize=15, fontweight='bold')
plt.ylabel('count', fontsize=15,fontweight='bold')
plt.show()


# In[12]:


# percentage of viewers types


#assign a fuction to the counts for easy vissualization
type_of_user = data.VisitorType.value_counts()


# In[13]:


fig, ax = plt.subplots(figsize=(10, 6))
x=type_of_user

labels = ['Returning_Visitor','New_Visistor','Other']

patches, texts, pcts = ax.pie(
    x, labels=labels, autopct='%.2f%%',
    wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'size': 'x-large'},
    startangle=180, explode =(0.1,0,0.2))
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='white')
plt.setp(texts, fontweight=600)
ax.set_title('Type of user', fontsize=18)
plt.tight_layout()


# ### About ten thousand five hundred and fifty-one persons which represent 85.6% of shopers were returning vissitors.
# Also One thousand six hundred and ninety-four persons which represent 13.7% of shopers were new vissitors.
# Lastly eighty-five persons which represents 0.7%  of shoppers belongs to other categories.
# 

# ## Revenue Analysis (whether or not a user completed the purchase)

# In[14]:


data['Revenue'].replace({False:'Not completed purchase',True: 'completed purhase'}, inplace=True)


# In[15]:


data.Revenue.value_counts()


# In[16]:


plt.figure(figsize=(8,8))
plt. title('Assesment of people wheather they completed their purchase or not', fontsize=15, fontweight='bold') 
ax = sns.countplot(x='Revenue', data = data)

for i in ax.containers:
    ax.bar_label(i)
plt.ylabel('counts', fontsize=15, fontweight='bold')
plt.xlabel('Categories', fontsize=15, fontweight='bold')
plt.show()


# In[17]:


fig, ax=plt.subplots(figsize=(8,7))

labels = ['not completed purchase','completed purchase']
x = [ 10422,1908]

patches,texts,pct = ax.pie(x, labels=labels, autopct ='%.2f%%',
                          wedgeprops={'linewidth':3.0, 'edgecolor':'white'},textprops={'size':'x-large'},
                           explode=(0,0.1),
                          startangle=90)
for i, patch in enumerate(patches):
    texts[i].set_color(patch.get_facecolor())

plt.setp(texts, fontweight=200)
ax.set_title('% of people who did or did not complete their purchase', fontsize=18, fontweight='bold')
plt.setp(pct, color='white')
plt.tight_layout()
plt.show()


#  Ten thousand four hundred and twenty-two persons which represents about 85% of people did not complet their purchases when they visited the siteWhile One thousand nine hundred and eight people which reprsent 15% of persons completed their purchase when they visited the site

# #  weekends Analysis

# In[18]:


data['Weekend'].replace({False:'weekdays',True:'weekends'},inplace=True)


# In[19]:


plt.figure(figsize=(8,8))
plt. title('numbers of transactions that where done during weekends or weekdays',fontsize=15,fontweight='bold')
ax = sns.countplot(x='Weekend', data = data)
for i in ax.containers:
    ax.bar_label(i)
plt.ylabel('Counts',fontsize=15)
plt.xlabel('Either Weekends or Not',fontsize=15)
plt.show()


# In[20]:


week_days_analy = data.Weekend.value_counts(normalize = True)
display(week_days_analy)


# In[21]:


# percentage of transactions that where done on weekends or not


fig, ax=plt.subplots(figsize=(8,7))

labels = ['weekdays','weekends']
size = [ 9462,2868]

patches,texts,pct = ax.pie(size, labels=labels, autopct ='%.2f%%',
                          wedgeprops={'linewidth':3.0, 'edgecolor':'white'},textprops={'size':'x-large'},
                           explode=(0,0.1),
                          startangle=90)
for i, patch in enumerate(patches):
    texts[i].set_color(patch.get_facecolor())

plt.setp(texts, fontweight=500)
ax.set_title('Numbers of transactions that where done during weekends or weekdays in %', fontsize=18, fontweight='bold')
plt.setp(pct, color='white')
plt.tight_layout()
plt.show()


# A total number of nine thousand four hundred and sixty-two persons which represents 77% are categories of purchases that where done on week days While about Two thousand eight hundred sixty-eight which represents 23% where categories of purchases that where done on weekends.

# # The month Analysis
# 

# In[22]:


data.Month.value_counts()


# In[24]:


plt.figure(figsize=(10,8))
plt.title('Counts of months in which the pages of the sites where viewed', fontsize=18,fontweight='bold')
ax=sns.countplot(x='Month', data = data, order=data['Month'].value_counts().index)
for i in ax.containers:
    ax.bar_label(i)
plt.xlabel('months',fontsize=15)
plt.ylabel('Counts',fontsize=15)    
plt.show()


# In[29]:


# percentages of pages viewd by months

sizes = data.Month.value_counts()

fig, ax=plt.subplots(figsize=(8,7))

labels = ['May','Nov','Mar','Dec','Oct','Sept','Aug','Jul', 'Jun','Feb']
size = ['sizes']

patches,texts,pct = ax.pie(sizes, labels=labels, autopct ='%.2f%%',
                          wedgeprops={'linewidth':3.0, 'edgecolor':'white'},textprops={'size':'x-large'},
                          explode=(0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.4,0.1,0.1),startangle=360)
for i, patch in enumerate(patches):
    texts[i].set_color(patch.get_facecolor())



plt.setp(texts, fontweight=200)
ax.set_title('Counts of months in which the pages of the sites where viewed in %', fontsize=18, fontweight='bold')
plt.setp(pct, color='white')
plt.tight_layout()
plt.show()


# The month of may seems to take the lead with the highest figure of page viewers followed by november, march came third and december follow suit. then the rest of the months. The least is febuary with just 184 numbers of persons who viewed the page.

# # Special day anaysis.

# In[30]:


data.SpecialDay.value_counts()


# In[31]:


#  writing a funtion to creat a new colunm from the  special day column properly

def Proximity_to_event_date (x):
    if x == 0.0:
        return 'Not close'
    elif x == 0.6:
        return '6 days close'
    elif x == 0.8:
        return '8 days close'
    elif x == 0.4:
        return '4 days close'
    elif x == 0.2:
        return '2 days close'
    else:
        return 'A day close'
        

# to apply the function above to the dataframe thereby creating a new column

data['ProximityToEventDate'] = data['SpecialDay'].apply(Proximity_to_event_date)

data
        


# In[32]:


# new shape of the data

data.shape


# In[ ]:


data.ProximityToEventDate.value_counts()


# In[33]:


# graphical representation of the proximity of a page viewer to an event date.

plt.figure(figsize=(10,8))

ax= sns.countplot(x = 'ProximityToEventDate', data = data)
for j in ax.containers:
    ax.bar_label(j)
plt.title('The proximity of a browser to a special day',fontsize=18,fontweight='bold')
plt.xlabel('proximity',fontsize=15)
plt.ylabel('counts',fontsize=15)


# In[34]:


plt.figure(figsize=(10,10))

data.ProximityToEventDate.value_counts().plot.pie(autopct ='%.2f%%')
plt.title('The proximity of a browser to an event date in %')
plt.show


#  About 11079 persons which comprises of 90% did not brows the sites  on a date close  to a special day. While about 154 persons which is just 1% of persons brows the site a day close to a special day.
# 
# There is a clear indication that majority of the pageviewers did not brows on dates close to special days, by implication the hypothesis or asumption of people completing their transactions when they visit the site is hereby rejected.

# In[35]:


# A cross section of visitors to the shoping mall sites by months and their proximity to an event dates

cross_sec_pivot = pd.crosstab(data['Month'],data['ProximityToEventDate'])
cross_sec_pivot


# In[36]:


plt.figure(figsize=(15,10))

ax = sns.countplot(x = 'Month',data = data, hue = 'ProximityToEventDate')
for i in ax.containers:
    ax.bar_label(i)
plt.title('Analysis of page viewers proximity by months',fontsize=18,fontweight='bold')
plt.xlabel('Months',fontsize=15)
plt.ylabel('counts of proximity',fontsize=15)
plt.show()


# The month of may seems to have a significant numbers of persons scattered across board who visited the online shopping site; about 2192 persons who vissited were not close to any event dates, 222 persons were four days close, 306 persons were eight days close, one-fourty-nine persons were a day close, one-sixty-three were two days close while 332 persons were six days close.
# 
# febuary also has a reasonable numbers of persons scattered across board; 15 persons vissited the sites two days close to events dates, 21 persons were four days close, 19 were six days close, another 19 were eight days close, 5 persons were a day close while about one-hundred and five were not close at all.
# 
# Other months comprises majorly of persons who vissited the sites on days not close to any events dates at all.

# In[37]:


# posibility of completing a transaction based on proximity to an events date

pro_of_posibility_Pivot = pd.crosstab(data['ProximityToEventDate'],data['Revenue'])
pro_of_posibility_Pivot                                  


# In[38]:


plt.figure(figsize=(14,8))

ax=sns.countplot(x='ProximityToEventDate', hue='Revenue', data=data)
for i in ax.containers:
    ax.bar_label(i)
plt.title('Distribution of proximity to event dates and whether or not a user completed the purchase',
          fontweight='bold',fontsize=18)
plt.xlabel('proximity to event dates', fontsize=15)
plt.ylabel('counts', fontsize=15)
plt.show()


# ### About 10,422 persons viewed the site and  where not able to complet a purchase,while about 1,908 who also viewed the sites were able to complet a purchase.
# 
# ### out of this numbers of people who viewd the sites on a day not close to any event dates, 9,248 persons were not able to complete any purchase, while 1,831 were able to complete a purchase, 
# ### furthermore; those who viewed the sites four days close to an events dates, 230 were not able to complete a purchase while 13 persons were able to complete a purchase, 
# 
# ### Eight days close; 314 were not able to complet a purchase while 11 persons completed their purchase, 
# 
# ### Two days close; 164 were not able to complete a purchase while 10 persons completed their purchase.
# 
# ### Six days close 322 were not able to complete their purchase while 29 persons completed theirs. 
# 
# ### A day close;144  were not able to complet a purchase while  10 completed their purchase.

# In[39]:


unique_values = data['Region'].sort_values().unique()
print(unique_values)


# In[40]:


data.Region.value_counts().sort_values(ascending = True)


# In[41]:


plt.figure(figsize=(10,8))
plt.title('Count of people who are viewing from same location', fontsize=18,fontweight='bold')

ax=sns.countplot(x='Region', data = data,  order=data['Region'].value_counts().index)

for i in ax.containers:
    ax.bar_label(i)
plt.xlabel('Regions', fontsize=15)
plt.ylabel('count', fontsize=15)    
plt.show()


# In[42]:


plt.figure(figsize=(10,10))

data.Region.value_counts().plot.pie(autopct ='%.2f%%')
plt.title('% of people who viewed from a particular region',fontsize=18,fontweight='bold')
plt.show


# There seems to exist about nine different locations or regions where people where viewing the page from. region one recorded the highest figure followed by region three, followed by two and four as well. while the 5th regions has the least page viewers.¶

# ### ProductRelated analysis

# In[43]:


# viewing the numbers of productrelated pages

pro_rel_page_uniq = data['ProductRelated'].sort_values().unique()
print(pro_rel_page_uniq)


# In[44]:


page_counts = data['ProductRelated'].value_counts()

data.ProductRelated.value_counts()


# In[45]:


plt.figure(figsize=(12,8))

sns.histplot(data['ProductRelated'])
plt.title('productrelated pages vissited by a user',fontsize=18,fontweight='bold')
plt.xlabel('productrelated pages')
plt.ylabel('frequency')
plt.show()


# In[46]:


plt.figure(figsize=(8,8))

sns.boxplot(data = data, y='ProductRelated')
plt.show()


# There are about seven-hundred and five product related pages, and most of the page vissitors vissited the first fifty to one-hundred pages.
# 

# # Bounce Rates Analysis

# In[47]:


plt.figure(figsize=(10, 6))
plt.hist(data['BounceRates'], bins=20, edgecolor='black')  
plt.title('Bounce Rates Distribution',fontweight='bold', fontsize=18)
plt.xlabel('Bounce Rates')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[48]:


plt.figure(figsize=(8,6))

sns.boxplot(y='BounceRates', data=data)
plt.show()


# The distribution in the histogram above is skewed to the left (negative distribution), which means that a large numbers of persons intered the sites and exited without trigering any further action

# ### ExitRates Analysis

# In[49]:


plt.figure(figsize=(10, 6))
plt.hist(data['ExitRates'], bins=20, edgecolor='black')  
plt.title('Bounce Rates Distribution', fontweight='bold',fontsize=18)
plt.xlabel('Exit Rates')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# up to 75% of persons vissited the shoping site and end up in one page without trIgering any further action.

# # Feature Engineering

# In[50]:


data.columns


# In[51]:


data


# In[52]:


# to transform the target column 
# import the library 

from sklearn.preprocessing import LabelEncoder


# In[53]:


# import the library 
le = LabelEncoder()


# In[54]:


# encode the target column 'income'

data['Revenue'] = le.fit_transform(data['Revenue'])
data


# In[55]:


data['Weekend'] = le.fit_transform(data['Weekend'])
data


# ## Using one-hot encoding to convert the categorical columns

# In[56]:


Categorical = ['VisitorType', 'ProximityToEventDate',  ]


# In[57]:


Categories_dummies = pd.get_dummies(data[Categorical])


Categories_dummies.head()


# In[58]:


# inculcate the categories dumies to the original dataframe

data = pd.concat([data, Categories_dummies], axis=1)


print(data.shape)
data.head()


# In[59]:


# droping the original categorical coulmn


data = data.drop(Categorical,axis=1)
data


# In[60]:


# drop the month column its of no use in the analysis

data.drop(['Month'],axis=1,inplace=True)
data


# In[61]:


data.shape


# In[62]:


# choosing the target

y = data.Revenue
x = data.drop('Revenue', axis=1)


# In[63]:


print(x.head())


# In[64]:


print(y.head())


# In[65]:


# import the necessary model for predictions

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix


# In[102]:


model = LogisticRegression(max_iter=1000)


from sklearn.preprocessing import StandardScaler

# Scale the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[103]:


# initialize the models

LR = LogisticRegression()
DT = DecisionTreeClassifier()
KN = KNeighborsClassifier()


# In[104]:


# creating a list of the models

models = [LR,KN,DT]


# In[105]:


# spiting of the data into 30% ratio (0.30)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[112]:


#create function to train the models and evaluate accuracy
def trainer(models,X_train,y_train,X_test,y_test):
    #fit your model
    model.fit(X_train,y_train)
    #predict on the fitted model
    prediction = models.preds(X_test)
    #print evaluation metric
    print('\nFor {}, Accuracy score is {} \n'.format(model.__class__.__name__,accuracy_score(prediction,y_test)))
    print(classification_report(prediction, y_test))
    print(confusion_matrix(prediction,y_test))
    


# In[113]:


import warnings
warnings.filterwarnings("ignore")


# In[114]:


#loop through each model, training in the process

for model in models:
    model.fit(x_train, y_train)
    # Predict on the fitted model
    prediction = model.predict(x_test)
    # Print evaluation metric
    print('\nFor {}, Accuracy score is {}\n'.format(model.__class__.__name__, accuracy_score(prediction, y_test)))
    
    print(classification_report(prediction, y_test))
    print(confusion_matrix(prediction,y_test))


# # SUMMARY OF FINDINGS AND RECOMENDATIONS:
# 
# 
# There are three unique sets of people who vissited this online shopping sites, i.e (returning vissitors)those who vissited and later came back, the first time vissitors, and the other categories. The records covers a period of ten months, with just january and April been omitted from the months in the year. 
#  
#  About 85.57% people were among those who vissited and later came back the second time, while 13.74% of people were new vissitors and 0.69 covers the other categories. Furthermore, about 85% of the sites vissitors did not completes their purchase or transactions while just 15% of people completes their transactions or purchase from the online shopping sites. About 77% of the transactions were done during week-days while 23% of the transactions were don on week-ends.
#   
#  During this periods of this ten months, the month of May and November has the highest numbers of people who checked on products within the sites, A further analysis was conducted to see the proximity of the shoppers to a special day like mothers day, vallentine etc. this reveal that about 90% of the shoppers did not vissited the sites on days close to any special day, while the remaining 10% was allocated to 8-days, 6-days, 4-days, 2-days and A-day close to a special day. An average of 50% of people vissited the product related pages on the sites, while an average of 25% of people vissited without trigering any further action. 
#       
#       
#    ### Three machine learning algorithm were used to predict the shoppers intention that is;
#    
#    LogisticRegression; This has a True positive of 3037, False Positive of 360, False Negative of 87 and a True Negative of 215 Acuracy of 88%, 
#    DecisionTreeClassifier; This has a True Positive of 2826, False Negative of 258, False Positive of 298 and a True Negative of 317. Acuracy of 85%
#    KNeighborsClassifier; This has a True positive of 3004, False negative of 420, False positive of 120 and a True negative of 155. Acuracy of 85%
# 
# 
# 

# In[ ]:





# In[ ]:




