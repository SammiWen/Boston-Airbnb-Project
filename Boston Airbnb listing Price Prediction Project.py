#!/usr/bin/env python
# coding: utf-8

# ### Analytics Plan
# #### Step1:  Import Dataset and Prepare Data
#       Step1.1:Missing Value data check
#       Step1.2:Data Type check
#       Step1.3:Numarical data type check (range)
#       Step1.4:Categorical data type check
# #### Step2: Data Cleanning
# #### Step3: EDA
#        Step3.1: Numerical Features Analysis
#        Step3.2: Deal with the categorical Variables
# #### Step4: Prepard data for model building
#        Step4.1: Response variables -- Outliner Treatment
#        Step4.2: Numarical variables -- Multicollinearity Check
#        Step4.3: Categorical variables -- relative analysis and dummy variables 
# #### Step5: Build linner regression modle
# #### Step6: Conduct factor important analysis

# ### Step1: Import Dataset and Prepare Data

# In[1]:


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


# ### Import Dataset

# In[2]:


listings_raw = pd.read_csv('listings.csv', index_col = "id")

listings = listings_raw.copy()


# In[3]:


listings.head()


# ## Prepare Data

# ### Rows:3585, Columns:94

# In[4]:


num_rows = listings.shape[0] #Provide the number of rows in the dataset
num_cols = listings.shape[1] #Provide the number of columns in the dataset


# In[5]:


num_rows


# In[6]:


num_cols


# ### Step1.1: Missing Value data check
# #### 50 variables has no missing value

# In[7]:


no_nulls = set(listings.columns[~listings.isnull().any()])
len(no_nulls)


# #### 9 Variables has more than half missing value

# In[8]:


most_missing_cols = set(listings.columns[listings.isnull().mean() > 0.50])
most_missing_cols


# ### Step1.2:Data Type check

# In[9]:


# Summary:
#1. 4 varialbes have 0 non-null value (neighbourhood_group_cleansed,has_availability,license,jurisdiction_names)
#2. Some columns relative to $ need to change to number data tpye
#3. Some columns can be removed from the data like, listing_url,scrape_id,last_scraped
print(listings.info())


# #### Step1.3:Numarical data type check ( Remember check the range)
# **1.** Variables scrape_id', 'host_id','jurisdiction_names'are not useful for model building
# **2.** Variables maximum_nights and availability_365 have high standard diviation which will afact the model accuracy. 

# In[10]:


num_vars = listings.select_dtypes(include=['float', 'int']).columns
num_vars


# In[11]:


pd.set_option('display.max_columns', None)
listings.select_dtypes(include=['float', 'int']).describe()


# #### Step1.4: Categorical data type check
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

# In[12]:


cat_vars = listings.select_dtypes(include=['object']).columns
cat_vars


# ### Step2: Data Cleanning

# ####  Question 1: What's the difference between superhost and regular host? (How to analyze mutiple chosse question?)
# #### Question 1.1: Do super hosts respond faster than the regular host?

# In[13]:


# Supperhost did respond faster than the regular host
response_time_superhost=pd.crosstab(listings.host_response_time, listings.host_is_superhost)
response_time_superhost/response_time_superhost.sum()


# #### Internet is the most popular amenities, while wheelchair Accessible is provided by least of the house owner

# In[14]:


# Check the options in the answer
listings['amenities'].value_counts()


# In[15]:


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


# In[18]:


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


# In[19]:


possible_vals =["TV","Cable TV","Internet","Wireless Internet","Air Conditioning","Wheelchair Accessible",
                "Kitchen,Doorman","Elevator in Building","Buzzer/Wireless Intercom","Heating","Family/Kid Friendly",
                "Washer,Dryer","Smoke Detector","Carbon Monoxide Detector","First Aid Kit","Safety Card",
                "Fire Extinguisher","Essentials","Shampoo","24-Hour Check-in","Hangers","Hair Dryer",
                "Laptop Friendly Workspace"]

props_df = clean_and_plot(listings,'amenities','Amenities Distribution')


# #### Question 1.2: Which amenities are more likely to be provided by the superhost vs. the regular host?
# If you choose Super host, you will be more likely to get Shampoo,First Aid Kit,Carbon Monoxide Detector than the none superhost

# In[20]:


superhost_or_not=listings.host_is_superhost.value_counts()
superhost_or_not


# In[21]:


def superhost_or_not(formal_ed_str):

    if formal_ed_str in("t"):
        return (1)
    else:
        return (0)

listings["host_is_superhost"].apply(superhost_or_not)[:5] #Test your function to assure it provides 1 and 0 values for the df


# In[22]:


listings['Superhost'] = listings["host_is_superhost"].apply(superhost_or_not)


# In[23]:


ed_1 = listings[listings['Superhost']== 1] # Subset df to only those with HigherEd of 1
ed_0 = listings[listings['Superhost']== 0] # Subset df to only those with HigherEd of 0
print(ed_1['Superhost'][:5]) #Assure it looks like what you would expect
print(ed_0['Superhost'][:5]) #Assure it looks like what you would expect


# In[24]:


#Check your subset is correct - you should get a plot that was created using pandas styling
#which you can learn more about here: https://pandas.pydata.org/pandas-docs/stable/style.html

ed_1_perc = clean_and_plot(ed_1,'amenities','Superhost',plot=False)
ed_0_perc = clean_and_plot(ed_0,'amenities','Normal',plot=False)

comp_df = pd.merge(ed_1_perc, ed_0_perc, left_index=True, right_index=True)
comp_df.columns = ['Superhost', 'None-Superhost']
comp_df['Diff_HigherEd_Vals'] = comp_df['Superhost'] - comp_df['None-Superhost']
comp_df.style.bar(subset=['Diff_HigherEd_Vals'], align='mid', color=['#d65f5f', '#5fba7d'])


# #### Question 1.3: Do the superhosts' review score is higher than the none superhost?

# In[27]:


listings.groupby("Superhost", as_index=True)['review_scores_rating','review_scores_accuracy','review_scores_checkin', 
                                       'review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                                       'review_scores_location','review_scores_value'].mean() 


# #### Question 2: Is there any bias between the house or other house tpye among the Amenity type ?
# If you choose house, you are more likely to have TV, Cable TV, Family/Kid Friendly	

# In[28]:


room_type=listings.room_type.value_counts()
room_type


# In[29]:


def house (formal_ed_str):

    if formal_ed_str in("Entire home/apt"):
        return (1)
    else:
        return (0)

listings["room_type"].apply(house)[:5] #Test your function to assure it provides 1 and 0 values for the df


# In[30]:


listings['house_or_not'] = listings["room_type"].apply(house)


# In[31]:


ed_house = listings[listings['house_or_not']== 1] # Subset df to only those with HigherEd of 1
ed_not_house = listings[listings['house_or_not']== 0] # Subset df to only those with HigherEd of 0
print(ed_house['house_or_not'][:5]) #Assure it looks like what you would expect
print(ed_not_house['house_or_not'][:5]) #Assure it looks like what you would expect


# In[32]:


#Check your subset is correct - you should get a plot that was created using pandas styling
#which you can learn more about here: https://pandas.pydata.org/pandas-docs/stable/style.html

ed_1_perc = clean_and_plot(ed_house,'amenities', 'House', plot=False)
ed_0_perc = clean_and_plot(ed_not_house,'amenities', 'Other', plot=False)

comp_df = pd.merge(ed_1_perc, ed_0_perc, left_index=True, right_index=True)
comp_df.columns = ['House', 'Others']
comp_df['Diff_HigherEd_Vals'] = comp_df['House'] - comp_df['Others']
comp_df.style.bar(subset=['Diff_HigherEd_Vals'], align='mid', color=['#d65f5f', '#5fba7d'])


# ### Data Cleanning_make the funtion resusable
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

# In[33]:


def clean_dataset (df, no_usage,response_variable,Money_Variable,Transfer_Variables,Time_Length,till_date,count_items):
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
    num_vars = df_dropna.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df_dropna[col].fillna((df_dropna[col].mean()), inplace=True)
        
    x=df_dropna
    return x
    


# In[34]:


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


# In[35]:


x1.head()


# #### Question3: What's the top 10 expensive neighbourhood?
# Downtown,Beacon Hill,Back Bay,South End,North End,South Boston Waterfront,Leather District,South Boston,West End,Fenway is the top 10 expensive neighbourhood

# In[37]:


x1.groupby(['neighbourhood_cleansed']).mean()['price'].sort_values()


# ### Step3: EDA
# #### Step3.1:Numerical Features Analysis

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

# In[38]:


corrlation = x1.select_dtypes(include=['int64', 'float64']).corr()
mask = np.zeros_like(corrlation)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(24,12))
plt.title('Heatmap of corr of features')
sns.heatmap(corrlation, mask = mask, vmax=.3, square=True, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# 
# 

# #### Step3.2: Deal with the categorical Variables

# In[39]:


# Check the categorical variables again
x1.select_dtypes(include=['object']).columns


# #### More than 40% have strict cancellation policy

# In[40]:


cancellation_policy_count=x1.cancellation_policy.value_counts()
(cancellation_policy_count/x1.shape[0]).plot(kind="bar");
plt.title("cancellation_policy_count");


# #### Step3.2 Choose Categorical Variable 

# #### The price among neighbourhood ,property_type, room_type,require_guest_phone_verification,bed_type have obvious price difference

# In[41]:


x1.groupby(['zipcode']).mean()['price'].sort_values().dropna()


# In[42]:


x1.groupby(['is_location_exact']).mean()['price'].sort_values().dropna()


# In[43]:


x1.groupby(['property_type']).mean()['price'].sort_values().dropna()


# In[44]:


x1.groupby(['room_type']).mean()['price'].sort_values().dropna()


# In[45]:


x1.groupby(['requires_license']).mean()['price'].sort_values().dropna()


# In[46]:


x1.groupby(['instant_bookable']).mean()['price'].sort_values().dropna()


# In[47]:


x1.groupby(['cancellation_policy']).mean()['price'].sort_values().dropna()


# In[48]:


x1.groupby(['require_guest_profile_picture']).mean()['price'].sort_values().dropna()


# In[49]:


x1.groupby(['require_guest_phone_verification']).mean()['price'].sort_values().dropna()


# In[50]:


x1.groupby(['host_is_superhost']).mean()['price'].sort_values().dropna()


# In[51]:


x1.groupby(['host_response_time']).mean()['price'].sort_values().dropna()


# In[52]:


x1.groupby(['host_has_profile_pic']).mean()['price'].sort_values().dropna()


# In[53]:


x1.groupby(['host_identity_verified']).mean()['price'].sort_values().dropna()


# In[54]:


x1.groupby(['bed_type']).mean()['price'].sort_values().dropna()


# In[55]:


x1.groupby(['neighbourhood_cleansed']).mean()['price'].sort_values().dropna()


# In[56]:


x1.groupby(['host_identity_verified']).mean()['price'].sort_values().dropna()


# In[57]:


x1.groupby(['host_has_profile_pic']).mean()['price'].sort_values().dropna()


# In[58]:


x1.groupby(['host_is_superhost']).mean()['price'].sort_values().dropna()


# ### Step4: Prepard data for model building
# Step4.1: Response variables -- Outliner Treatment

# #### Response variable -- Outliner Treatment

# In[59]:


print(x1['price'].describe())


# In[60]:


# visualizae the price
plt.figure(figsize=(8, 6))
sns.distplot(x1['price'], bins=50, kde=True)
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Price (dollar)', fontsize=12)
plt.title('Listed Price Distribution', fontsize=14);


# In[61]:


# remove outliers
#Calculate the outliners Q2 + 1.5*IRQ
#Outliner_right= 220+1.5* (220-85) =423
x2 = x1[x1['price'] < 423]
x2.shape


# In[62]:


plt.figure(figsize=(8, 6))
sns.distplot(x2['price'], bins=50, kde=True)
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Price (dollar)', fontsize=12)
plt.title('Listed Price Distribution after remove outlinner', fontsize=14);


# ####  Step4.2: Numarical variables -- Multicollinearity Check

# In[63]:


def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[64]:


# Many of the variables are highly corelative
X_VIF = x2.select_dtypes(include=['int64', 'float64']).drop(['price'], axis=1)
calc_vif(X_VIF)


# In[65]:


VIF_test_col =[ 'bathrooms','cleaning_fee','guests_included', 'minimum_nights',
               'number_of_reviews','reviews_per_month','host_listings_count',
              'amenities','extra_people','availability_30','accommodates']
X_vif= x2[VIF_test_col]
calc_vif(X_vif)


# #### Step4.3: Categorical variables -- relative analysis and dummy variables 

# In[66]:


clean_data= x2[[ 'property_type', 'room_type','require_guest_phone_verification','bed_type','bathrooms',
       'cleaning_fee','guests_included', 'minimum_nights','number_of_reviews','reviews_per_month',
       'host_listings_count','amenities','extra_people','availability_30','accommodates','price']]
clean_data.head()


# In[67]:


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


# In[68]:


proccessed_data=create_dummy_df(clean_data, dummy_na =True)
proccessed_data.head()


# In[69]:


list(proccessed_data.columns)


# ### Step5: Build linner regression modle

# In[70]:


#Split into explanatory and response variables
y_raw = np.log2(proccessed_data['price'].astype(int))
y_raw = np.log2(proccessed_data['price'].astype(int))
x_raw= proccessed_data.drop(['price'],axis=1)
#Split into train and test
X1_train, X1_test, y1_train, y1_test = train_test_split(x_raw, y_raw, test_size = .30, random_state=42)
lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X1_train, y1_train) #Fit

#Predict and score the model
y_test_preds = lm_model.predict(X1_test) 
y_train_preds = lm_model.predict(X1_train) 

#Rsquared and y_test
rsquared_score = r2_score(y1_test, y_test_preds)#r2_score
length_y_test = len(y1_test)#num in y_test

train_score = r2_score(y1_train, y_train_preds)

"The r-squared score for your model was {} on {} values,train set r-squared was{} .".format(rsquared_score, length_y_test,train_score)


# ### Step6: Conduct factor important analysis
# Variables room type,property_type,bed_type,property_type_Bed & Breakfast,require_guest_phone_verification_t,accommodates afact the price most.

# In[71]:



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
coef_df = coef_weights(lm_model.coef_, X1_train)

#A quick look at the top results
coef_df.head(20)


# ### Conclusion:
# ### Summary*******************************************************************************************************************
# #### Q1. What's the difference between superhost and regular host?
# #### 1) Do super hosts respond faster than the regular host? 
#       
#       Yes. The proportion of superhost who respond within an hour is 1.3x higher than the none regular host.
#    
# ####  2) Which amenities are more likely to be provided by the superhost vs. the regular host?
#       
#       Air Conditioning, shampoo, Fire extinguishers, and Carbon Monoxide Detectors.
#       
# ####  3) Do the superhosts' review score is higher than the none superhost?
#       The superhost's satisfaction is higher than the regular host, especially in value and cleanliness.
#       
# #### Q2. Is there any bias between the house or other house tpye among the Amenity type ?
#    Yes,houses are more likely have TV, Family/Kid Friendly and Cable TV.
# #### Q3. What's the top 10 expensive neighbourhood?
#    Downtown,Beacon Hill,Back Bay,South End,North End,South Boston Waterfront,Leather District,South Boston,West End,Fenway is the top 10 expensive neighbourhood
# #### Q4.  What's the top 5 factors afact the listing price?
#    Variables room type,property_type,bed_type,property_type_Bed & Breakfast,require_guest_phone_verification_t,accommodates afact the price most
# 
# 
# 
# 
