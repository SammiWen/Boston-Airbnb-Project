#!/usr/bin/env python
# coding: utf-8

# ### Project: Write a Data Science Blog Post
# ### Part 1: Business Understanding
# #### Brief Description
#      Airbnb has become one of the most popular platforms to choose a dream vacation living place. 
#      I used data from Kaggle Boston Airbnb Open Data to look at the questions as below. 
#      Question 1: What are the top 5 factors that affect the listing price? 
#      Question 2: What's the difference between superhost and regular host?
#              2.1 Do super hosts respond faster than the regular host?
#              2.2 Which amenities are more likely to be provided by the superhost vs. the regular host?
#              2.3 Do the superhosts' review score is higher than the none superhost?
#      Question 3: what are the top 10 expensive neighborhoods? 
# ### Part 2: Data Understanding
#      2.1: Missing Value and data tpye Check
#      2.2: Numerical Features Analysis
#      2.3: Categorical Features Analysis
# ### Part 3: Prepare Data
#      3.1: Response variables -- Outliner Treatment
#      3.2: Numarical variables -- Multicollinearity Check
#      3.3: Categorical variables -- relative analysis and dummy variables 
# ### Part 4: Data Modeling
#      4.1: Generate test design
#      4.2: Build the model
#      4.3: Assess model
# ### Part 5: Evaluate Result

# ### Part 2: Data Understanting

# In[1]:


# import the libaries needed in the project
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor #VIF Check

# check scikit-learn version
import sklearn
print(sklearn.__version__)


# #### Import Dataset

# In[2]:


listings_raw = pd.read_csv('listings.csv', index_col = "id")

listings = listings_raw.copy()


# In[3]:


listings.head()


# #### Data Szie Rows:3585, Columns:94

# In[4]:


num_rows = listings.shape[0] #Provide the number of rows in the dataset
num_cols = listings.shape[1] #Provide the number of columns in the dataset


# In[5]:


num_rows


# In[6]:


num_cols


# ### Step2.1.1: Missing Value data check
# #### 50 variables has no missing value

# In[7]:


no_nulls = set(listings.columns[~listings.isnull().any()])
len(no_nulls)


# #### 9 Variables has more than half missing value

# In[8]:


most_missing_cols = set(listings.columns[listings.isnull().mean() > 0.50])
most_missing_cols


# ### Step2.1.2:Data Type check

# In[9]:


# Summary:
#1. 4 varialbes have 0 non-null value (neighbourhood_group_cleansed,has_availability,license,jurisdiction_names)
#2. Some columns relative to $ need to change to number data tpye
#3. Some columns can be removed from the data like, listing_url,scrape_id,last_scraped
#4. 'host_response_rate', 'host_acceptance_rate' should be changed to float
print(listings.info())


# #### Step2.1.2:Numarical data type check ( Remember check the range)
# **1.** Variables scrape_id', 'host_id','jurisdiction_names'are not useful for model building
# 
# **2.** Variables maximum_nights and availability_365 have high standard diviation which will afact the model accuracy. 

# In[10]:


num_vars = listings.select_dtypes(include=['float', 'int']).columns
num_vars


# In[11]:


# Variables maximum_nights and availability_365 have high standard diviation
pd.set_option('display.max_columns', None)
listings.select_dtypes(include=['float', 'int']).describe()


# In[12]:


cat_vars = listings.select_dtypes(include=['object']).columns
cat_vars


# #### Data Cleanning
# **1**. Delete the variables have no help in model building
# 
# **2**. Delete the variables have all null value and missing value more than 50%
# 
# **3**. Change'price', 'weekly_price', 'monthly_price','security_deposit', 'cleaning_fee'into float data tpye
# 
# **4**. Count how many days since the date
# 
# **5**. Count the items in'amenities'and 'host_verifications'
# 
# **6**. Fill the numarical missing value with mean

# In[13]:


def clean_dataset (df, no_usage,response_variable,Money_Variable,Transfer_Variables,Time_Length,till_date,count_items):
    '''
    INPUT:
    df - pandas dataframe you want to clean
    no_usage - list of clolumns name you want to delete
    response_variable - the variable you want to predict
    Money_Variable - list of columns have $ sign
    Transfer_Variables - list of variables contain %
    Time_Length - list of variables you want to calculate how many days untill the till data
    till_date - The end day of your calculation
    count_items -list of columns have multiple answers in a question with , to seperate them
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
    1. without no usage columns
    2. variables contain $ and % sign will change to float
    3. calculate the number of days since one specific day
    4. count the number of answers within the multiple choice question
    5. the numarical missing value has been fill with mean
    '''
      
    #1 Delete the variables have no help in model building
    drop_no_usage= df.drop(no_usage, axis=1) 

    #1.1 drop the rows has missing value in the respones variable
    MissY_drop = drop_no_usage.dropna(subset=[response_variable],axis=0) 

    #1.2 delete the cloumns have missing value more than 0.5
    nan_cols = MissY_drop.columns[MissY_drop.isnull().mean() > 0.50]
    df_dropna = MissY_drop.drop(nan_cols, axis=1) 

    #3.1 Change $  into float
    for col in Money_Variable:
        # remove $ and comma from price, ignore na values so that we wont get any errors.
        df_dropna[col] = df_dropna[col].map(lambda p : p.replace('$','').replace(',',''), na_action='ignore')
        # convert cols to float type
        df_dropna[col] = df_dropna[col].astype(float)

    #3.2 Change %  into float
    for col in Transfer_Variables:
        df_dropna[col] = df_dropna[col].str.extract(r'(\d+)')
        df_dropna[col] = df_dropna[col].astype('float')
        
    #4. Count how many days since the date
    df_dropna[Time_Length] = pd.to_datetime(df_dropna[Time_Length])
    temp = pd.to_datetime(till_date)

    df_dropna['host_len'] = df_dropna.host_since.apply(lambda x: pd.Timedelta(temp-x).days)
    df_dropna = df_dropna.drop(Time_Length, axis=1)

    #5. Extract the number of item in one cell
    for col in count_items:
        df_dropna[col] = df_dropna[col].apply(lambda x: len(x.replace('{', '').                        replace('{', '').replace('"', '').split(',')))
  
    #6. fill numarical missing value with mean
    # Most of the missing values are about the characters of a proporty
    # the missing data didn't have time-series problem,they are gneral problem (Continuous), 
    # so I fill missing with mean
    num_vars = df_dropna.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df_dropna[col].fillna((df_dropna[col].mean()), inplace=True)
        
    x=df_dropna
    return x


# In[14]:


no_usage= ['scrape_id', 'host_id','listing_url', 'last_scraped', 'name', 'summary', 'space', 
           'description', 'experiences_offered', 'neighborhood_overview', 'transit', 'access', 'interaction', 
           'house_rules', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_name', 
           'host_location', 'host_about','state','market', 'smart_location', 'country_code', 'country',
           'calendar_updated','calendar_last_scraped', 'first_review', 'last_review','host_thumbnail_url', 'host_picture_url']

response_variable = 'price'

Money_Variable = ["cleaning_fee", "price", "extra_people"]

Transfer_Variables= ['host_response_rate', 'host_acceptance_rate']

Time_Length='host_since'

till_date='08/03/2021'

count_items=['amenities','host_verifications']


x1=clean_dataset (listings,no_usage, response_variable, Money_Variable,Transfer_Variables,Time_Length,till_date,count_items)


# ### Step2.2:Numerical Features Analysis

# #### Correlation matrix Summary
# **1**. accommodates,beds,bedrooms,cleaning_fee are high correlative with price
# 
# **2**. avaliability_30,60,90, host_listing count,host_total_listing count are highly correlative
# 
# **3**. accommodateds are highlly correlative with beds,bedrooms,bathroom,cleening fee, gust_inluded
# 
# **4**. guest include is highly correlative with extra_people
# 
# **5**. number of review is highly correlative with reviews per month
# 
# **6**. review relative variables are highly corelative.

# In[15]:


corrlation = x1.select_dtypes(include=['int64', 'float64']).corr()
mask = np.zeros_like(corrlation)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(24,12))
plt.title('Heatmap of corr of features')
sns.heatmap(corrlation, mask = mask, vmax=.3, square=True, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# 
# 

# #### Step2.3: Deal with the categorical Variables

# In[16]:


# Check the categorical variables again
x1.select_dtypes(include=['object']).columns


# #### Categorical data type check summary
# **1**. 'listing_url', 'last_scraped', 'name', 'summary', 'space',
#        'description', 'experiences_offered', 'neighborhood_overview', 'notes',
#        'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
#        'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_name',
#        'host_since', 'host_location', 'host_about','state','market', 'smart_location', 'country_code', 'country','calendar_updated','calendar_last_scraped', 'first_review', 'last_review',
#        are not useful for model building
#        
# **2**.'price', 'weekly_price', 'monthly_price','security_deposit', 'cleaning_fee'should be changed to float
# 
# **3**. the number of items should be counted in 'amenities'and 'host_verifications'

# #### Summary:The price among zip code, neighbourhood ,property_type, room_type,require_guest_phone_verification,bed_type have obvious price difference

# In[17]:


# a large gap among different zipcode
x1.groupby(['zipcode']).mean()['price'].sort_values().dropna()


# In[18]:


# no obvious differences are observed
x1.groupby(['is_location_exact']).mean()['price'].sort_values().dropna()


# In[19]:


# a large gap among different property type
x1.groupby(['property_type']).mean()['price'].sort_values().dropna()


# In[20]:


# a large gap among entire home/apt and other room type
x1.groupby(['room_type']).mean()['price'].sort_values().dropna()


# In[21]:


x1.groupby(['requires_license']).mean()['price'].sort_values().dropna()


# In[22]:


# no obvious differences are observed
x1.groupby(['instant_bookable']).mean()['price'].sort_values().dropna()


# In[23]:


# a large gap among different cancellation_policy
x1.groupby(['cancellation_policy']).mean()['price'].sort_values().dropna()


# In[24]:


# no obvious differences are observed
x1.groupby(['require_guest_profile_picture']).mean()['price'].sort_values().dropna()


# In[25]:


# a large gap between require_guest_phone_verification
x1.groupby(['require_guest_phone_verification']).mean()['price'].sort_values().dropna()


# In[26]:


# no obvious differences are observed
x1.groupby(['host_is_superhost']).mean()['price'].sort_values().dropna()


# In[27]:


# a large gap among different host_response_time
x1.groupby(['host_response_time']).mean()['price'].sort_values().dropna()


# In[28]:


# no obvious differences are observed
x1.groupby(['host_has_profile_pic']).mean()['price'].sort_values().dropna()


# In[29]:


# no obvious differences are observed
x1.groupby(['host_identity_verified']).mean()['price'].sort_values().dropna()


# In[30]:


# no obvious differences are observed
x1.groupby(['bed_type']).mean()['price'].sort_values().dropna()


# In[31]:


# a large gap among different neighbourhood_cleansed
x1.groupby(['neighbourhood_cleansed']).mean()['price'].sort_values().dropna()


# In[32]:


# no obvious differences are observed
x1.groupby(['host_identity_verified']).mean()['price'].sort_values().dropna()


# In[33]:


# no obvious differences are observed
x1.groupby(['host_has_profile_pic']).mean()['price'].sort_values().dropna()


# In[34]:


# no obvious differences are observed
x1.groupby(['host_is_superhost']).mean()['price'].sort_values().dropna()


# 

# ### Part3: Prepard data for model building
# #### Step3.1: Response variables -- Outliner Treatment

# #### Response variable -- Outliner Treatment

# In[35]:


# The min value is 10 seems problematic
print(x1['price'].describe())


# In[36]:


# visualizae the price
# the data skew to the right
plt.figure(figsize=(8, 6))
sns.distplot(x1['price'], bins=50, kde=True)
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Price (dollar)', fontsize=12)
plt.title('Listed Price Distribution', fontsize=14);


# #### The response variable skew to the right, it seems like a lot of outliners there

# In[37]:


# remove outliers
#Calculate the outliners Q2 + 1.5*IRQ
#Outliner_right= 220+1.5* (220-85) =423
x2 = x1[x1['price'] < 423]
x2.shape


# In[38]:


plt.figure(figsize=(8, 6))
sns.distplot(x2['price'], bins=50, kde=True)
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Price (dollar)', fontsize=12)
plt.title('Listed Price Distribution after remove outlinner', fontsize=14);


# After removing the outlinner, the shape of the response variables seems more reasonable. 

# ####  Step3.2: Numarical variables -- Multicollinearity Check_Choose numarical variables
# From the correlation matrix we know that the metrics about the availability, review score and room characters are highly correlative, if we want our model more statble we can check the VIF

# In[39]:


def calc_vif(X):
    '''
    Input:
    a list of column names you want to check the VIF
    Output:
    The VIF values for the chosen variables
    '''
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[40]:


# Most of the variables have high VIF
# Review Score and availability relative variable has very high VIF indicting that they will doing the same job in the model
X_VIF = x2.select_dtypes(include=['int64', 'float64']).drop(['price'], axis=1)
calc_vif(X_VIF)


# In[41]:


# After trying many combination, I found that this combination is in a good balance.
VIF_test_col =[ 'bathrooms','cleaning_fee','guests_included', 'minimum_nights',
               'number_of_reviews','reviews_per_month','host_listings_count',
              'amenities','extra_people','availability_30','accommodates']
X_vif= x2[VIF_test_col]
calc_vif(X_vif)


# #### Step3.3: Categorical variables -- relative analysis and dummy variables 

# In[42]:


# Create a new data frame with the choosen categorical and numarical variables
clean_data= x2[[ 'property_type', 'room_type','require_guest_phone_verification','bed_type','bathrooms',
       'cleaning_fee','guests_included', 'minimum_nights','number_of_reviews','reviews_per_month',
       'host_listings_count','amenities','extra_people','availability_30','accommodates','price']]
clean_data.head()


# In[43]:


#Pull a list of the column names of the categorical variables

def create_dummy_df(df, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values, if fasle fill 00
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for col in  cat_vars:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df


# In[44]:


proccessed_data=create_dummy_df(clean_data, dummy_na =True)
proccessed_data.head()


# In[45]:


list(proccessed_data.columns)


# ### Part 4: Data Modeling 
# #### 4.1&4.2: Build linner regression modle

# In[46]:


#Split into explanatory and response variables
y_raw = proccessed_data['price'].astype(int)
x_raw= proccessed_data.drop(['price'],axis=1)
#Split into train and test
X1_train, X1_test, y1_train, y1_test = train_test_split(x_raw, y_raw, test_size = .30, random_state=42)
lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X1_train, y1_train) #Fit

#Predict and score the model
y_test_preds = lm_model.predict(X1_test) 
y_train_preds = lm_model.predict(X1_train) 


# #### 4.3 Access the model

# In[47]:


#Rsquared and y_test
rsquared_score = r2_score(y1_test, y_test_preds)#r2_score
length_y_test = len(y1_test)#num in y_test

train_score = r2_score(y1_train, y_train_preds)

# Mean Square erro check
MSE = mean_squared_error(y1_train,y_train_preds)

"The r-squared score for your model was {} on {} values,MSE was{} .".format(rsquared_score, length_y_test,MSE)


# In[48]:


# residual check
y_test_preds = lm_model.predict(X1_test)

preds_vs_act = pd.DataFrame(np.hstack([y1_test.values.reshape(y1_test.size,1), y_test_preds.reshape(y1_test.size,1)]))
preds_vs_act.columns = ['actual', 'preds']
preds_vs_act['diff'] = preds_vs_act['actual'] - preds_vs_act['preds']

plt.plot(preds_vs_act['preds'], preds_vs_act['diff'], 'bo');
plt.xlabel('predicted');
plt.ylabel('difference');


# The residual distribution looks like a fan, we have 2 point has a big difference at prices 55 and 58

# In[49]:


preds_vs_act[preds_vs_act.preds <0]


# In[50]:


plt.plot(preds_vs_act['preds'], preds_vs_act['actual'], 'bo');
plt.xlabel('predicted');
plt.ylabel('actual'); #there appears a slight positive trend 


# The residual seems didn't spread evenly, Let's use Log method to build the model again

# In[51]:


# log the response variable
#Split into explanatory and response variables
y2_raw = np.log2(proccessed_data['price'].astype(int)) ## log the response variable
x2_raw= proccessed_data.drop(['price'],axis=1)
#Split into train and test
X2_train, X2_test, y2_train, y2_test = train_test_split(x2_raw, y2_raw, test_size = .30, random_state=42)
lm_model2 = LinearRegression(normalize=True) # Instantiate
lm_model2.fit(X2_train, y2_train) #Fit

#Predict and score the model
y2_test_preds = lm_model2.predict(X2_test) 
y2_train_preds = lm_model2.predict(X2_train) 

#Rsquared and y_test
rsquared_score_2 = r2_score(y2_test, y2_test_preds)#r2_score
length_y2_test = len(y2_test)#num in y_test

train_score_2 = r2_score(y2_train, y2_train_preds)

"The r-squared score for your model was {} on {} values,train set r-squared was{} .".format(rsquared_score_2, length_y2_test,train_score_2)


# In[52]:


# residual check after log--It seems like the residual spreed more evenly
y2_test_preds = lm_model2.predict(X2_test)

preds_vs_act2 = pd.DataFrame(np.hstack([y2_test.values.reshape(y2_test.size,1), y2_test_preds.reshape(y2_test.size,1)]))
preds_vs_act2.columns = ['actual', 'preds']
preds_vs_act2['diff'] = preds_vs_act2['actual'] - preds_vs_act2['preds']

plt.plot(preds_vs_act2['preds'], preds_vs_act2['diff'], 'bo');
plt.xlabel('predicted');
plt.ylabel('difference');


# ### Part5: Evaluate the result
# #### Question 1: What's the top 5 factors afact the listing price?
# Variables room type,property_type,bed_type,property_type_Bed & Breakfast,require_guest_phone_verification_t,accommodates afact the price most.

# In[53]:



def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(lm_model2.coef_, X2_train)

#A quick look at the top results
coef_df.head(20)


# #### Question 2: What's the difference between superhost and regular host?
# #### 2.1 Do super hosts respond faster than the regular host?
# #### Yes, Supperhost did response faster than the none super host

# In[54]:


# Supperhost did respond faster than the regular host
# None of the superhost response in a few days or more
response_time_superhost=pd.crosstab(listings.host_response_time, listings.host_is_superhost)
response_time_superhost/response_time_superhost.sum()


# #### 2.2 Which amenities are more likely to be provided by the superhost vs. the regular host?

# In[55]:


# Check the options in the answer
listings['amenities'].value_counts()


# In[56]:


# Create a function to count the number of elements in one option
def total_count (df, col1, col2, look_for):
    '''
    INPUT:
    df - the pandas dataframe you want to search
    col1 - the column name you want to look through
    col2 - the column you want to count values from
    look_for - a list of strings you want to search for in each row of df[col]
    OUTPUT:
    new_df - a datafram of each look_for with the count of how often it shows up
    '''
    from collections import defaultdict
    new_df = defaultdict(int)
    for val in look_for:
        for idx in range (df.shape[0]):
            if val in df[col1][idx]:
                new_df[val]+= int(df[col2][idx])
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns =[col1,col2]
    new_df.sort_values('count',ascending= False, inplace= True)
    return new_df


# In[57]:


def clean_and_plot(df,variable,title, plot=True):
    '''
    INPUT 
        df - a dataframe holding the CousinEducation column
        title - string the title of your plot
        axis - axis object
        plot - bool providing whether or not you want a plot back
        
    OUTPUT
        study_df - a dataframe with the count of how many individuals
        Displays a plot of pretty things related to the CousinEducation column.
    '''
    
    option_raw = df[variable].value_counts().reset_index()
    option_raw.rename(columns={'index': 'method', variable: 'count'}, inplace=True)
    option_new = total_count(option_raw, 'method', 'count', possible_vals)

    option_new.set_index('method', inplace=True)
    if plot:
        (option_new/df.shape[0]).plot(kind='bar', legend=None);
        plt.title(title);
        plt.show()
    props_df = option_new/df.shape[0]
    return props_df


# In[58]:


possible_vals =["TV","Cable TV","Internet","Wireless Internet","Air Conditioning","Wheelchair Accessible",
                "Kitchen,Doorman","Elevator in Building","Buzzer/Wireless Intercom","Heating","Family/Kid Friendly",
                "Washer,Dryer","Smoke Detector","Carbon Monoxide Detector","First Aid Kit","Safety Card",
                "Fire Extinguisher","Essentials","Shampoo","24-Hour Check-in","Hangers","Hair Dryer",
                "Laptop Friendly Workspace"]

props_df = clean_and_plot(listings,'amenities','Amenities Distribution')


# #### Internet is the most popular amenities, while wheelchair Accessible is provided by least of the house owner

# In[59]:


# Check the distribution of superhost & regular host, only 407 of them are superhost
superhost_or_not=listings.host_is_superhost.value_counts()
superhost_or_not


# In[60]:


# Create a funtion to transfer true or false into 1 and 0
def superhost_or_not(formal_ed_str):
    '''
    Input:
    formal_ed_str:column name you want to transfer
    output:
    if the value is "t", it will be  transfer to 1, else will be 0
    
    '''
    if formal_ed_str in("t"):
        return (1)
    else:
        return (0)


# In[61]:


#Test your function to assure it provides 1 and 0 values for the df
listings["host_is_superhost"].apply(superhost_or_not)[:5]


# In[62]:


# Create a new cloumn call super host
listings['Superhost'] = listings["host_is_superhost"].apply(superhost_or_not)


# In[63]:


ed_1 = listings[listings['Superhost']== 1] # Subset df to only those with HigherEd of 1
ed_0 = listings[listings['Superhost']== 0] # Subset df to only those with HigherEd of 0
print(ed_1['Superhost'][:5]) # validation
print(ed_0['Superhost'][:5]) # validation


# In[64]:


ed_1_perc = clean_and_plot(ed_1,'amenities','Superhost',plot=False) # apply the funtion clean_and_plot for Superhost
ed_0_perc = clean_and_plot(ed_0,'amenities','Normal',plot=False) # apply the funtion clean_and_plot for regular host


# In[65]:


comp_df = pd.merge(ed_1_perc, ed_0_perc, left_index=True, right_index=True) # merge superhosst and 
comp_df.columns = ['Superhost', 'Regular_host']
comp_df['Diff_HigherEd_Vals'] = comp_df['Superhost'] - comp_df['Regular_host']
comp_df.style.bar(subset=['Diff_HigherEd_Vals'], align='mid', color=['#d65f5f', '#5fba7d'])


# If you choose Super host, you will be more likely to get Shampoo,First Aid Kit,Carbon Monoxide Detector than the none superhost

# #### 2.3 Do the superhosts' review score is higher than the none superhost?

# In[66]:


listings.groupby("Superhost", as_index=True)['review_scores_rating','review_scores_accuracy','review_scores_checkin', 
                                       'review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                                       'review_scores_location','review_scores_value'].mean() 


# The superhost's satisfaction is higher than the regular host, especially in value and cleanliness.

# #### Question 3: what are the top 10 expensive neighborhoods? 

# In[67]:


x1.groupby(['neighbourhood_cleansed']).mean()['price'].sort_values().dropna()


# Downtown,Beacon Hill,Back Bay,South End,North End,South Boston Waterfront,Leather District,
# South Boston,West End,Fenway is the top 10 expensive neighbourhood
