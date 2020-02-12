# In Jupyter notebook, while writing anything hit tab for getting more info about autocomplete
# also for parameters, hit shift+tab , twice as well, thrice as well
# ??function name for source code
# h for shortcuts in jupyter notebook

#check types
import types
a = 7.9
type(a)==types.FloatType #true
type(a)==types.IntType #false

#Array vs list
Array has elements with the same DT, list all the elements can be of different dt

#Imp functions in string
split()
strip()\

#sorted function with keys -- it always returns a list
sorted(data,key=lambda x: x["price"],reverse=True)

# *args and ****kwargs
def myFun(*argv):  
    for arg in argv:  
        print (arg) 

def myFun(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 
        
#List
alist=[1,2,3,4,5]
asquaredlist=[i*i for i in alist]

#Text file read
hamletfile=open("hamlet.txt")
hamlettext=hamletfile.read()
hamletfile.close()
hamlettokens=hamlettext.split()#split with no arguments splits on whitespace
len(hamlettokens)

OR

with open("hamlet.txt") as hamletfile:
    hamlettext=hamletfile.read()
    hamlettokens=hamlettext.split()
    print len(hamlettokens)

#Dictionaries
a={'one':1, 'two': 2, 'three': 3}
print a.keys()
print a.values()
for k,v in a.items():
    print k,v

#csv to DataFrame
df=pd.read_csv("all.csv", header=None, names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'])
df.head()
df.dtypes
print movies.describe()
pd.value_counts(df['column']) #histogram count of the series, non null ones
df.apply(pd.value_counts) #**histogram of all values in all columns
df.shape
df.columns
print(type(df.rating), type(df))
pd.groupby('year').rating.mean() # Analyse the ratings by year(a category)
pd.groupby('year').mean() #Analyse the mean for all numeric columns
pd.groupby('year').rating.mean().plot()
ratings.groupby('user_id').rating.agg(['mean','max','count','min']) #** groupby aggregate
df.rating < 3
np.sum(df.rating > 3) -- find count of how many more than 3
np.mean(df.rating < 3.0)   OR   (df.rating < 3).mean()
df.query("rating > 4.5")  OR   df[df.rating >4.5]# like a where clause
df[df.year < 0]
df[(df.year < 0) & (df.rating > 4)]
df[df.year.isnull()]

#change datatypes
df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int)

#Imp functions to use
df.rating.hist();

#We can construct another list by using the syntax below, also called a list comprehension.
asquaredlist=[i*i for i in alist]
asquaredlist

#Difference between numpy array vs list: Vectorization : operations on numpy arrays, and by extension, Pandas Series, are vectorized. You can add two numpy lists by just using + whereas the result isnt what you might expect for regular python lists. To add regular python lists elementwise, you will need to use a loop:

    
# numpy array to csv
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

df.describe(include='all')
train_df.describe(include=['O']) #categorical variables
train_store.Open.describe() #
df.info()
df.columns
df['ColName'].fillna(df['ColName'].median(), inplace = True)
x = df.values #convert DF to np array

# items per category
frame['cluster'].value_counts()


# unique
df['Store'].unique()
df.reset_index() #convert index to columns

# Data understanding
liver.groupby('Outcome').mean()

#open file for UnicodeDecodeError: 'utf-8' codec can't decode byte error
open('u.item', encoding = "ISO-8859-1")

# to show all the columns in spyder/jupyter in pandas and numpy/
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
numpy.set_printoptions(threshold=numpy.nan)
pd.options.display.max_colwidth = 100 # display full strings in dataframe
OR try this:
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', None)
sum(users['sex'] == 'M') -- count of males

#Import website html
from urllib.request import urlopen
url = 'http://www.crummy.com/software/BeautifulSoup'
source = urlopen(url).read()
print (source)
## count occurences of 'Soup'
print source.count('Soup')

#BeautifulSoup
import bs4 #this is beautiful soup
soup = bs4.BeautifulSoup(source)
print soup
print soup.prettify()
soup.findAll('a') #show how to find all a tags
link_list = [l.get('href') for l in soup.findAll('a')] ## get all links in the page
link_list = [l for l in link_list if l is not None and l.startswith('http')] # by removing the nulls from the result above

#Checking the skew of the train_data
print(complete.skew()) #check width spread wrt bell curve
complete.kurt() # check the height spread wrt bell curve

####ECDF To get the first impression about continious variables in the data we can plot ECDF.
cdf = ECDF(train['Sales'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('Sales'); plt.ylabel('ECDF');
#####

#change value in column based on other columns
train.loc[train.PromoInterval == 0, 'Interval'] = 'Zero'

# skewnss: +ve for +ve skew, 0 for no skew:  for +ve skew mean>median
df.skew()

# closed stores
train[(train.Open == 0) & (train.Sales == 0)].head()

#number of nulls nan
store.isnull().sum()

#unique values in df
df.Store.unique()

#Get row based on null values
df_train[df_train['Embarked'].isnull()]

# replace NA's by 0
store.fillna(0, inplace = True)

#inner join
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

#groupby
train_store.groupby('StoreType')['Sales'].describe()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() #to see which feature is related to output

grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()


##### sales trends factorplot
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = c) 
######

##### categorical to numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["male","female"])
le.transform(df_train['Sex'])
########

######### heatmap : use only for continuous variables
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()
##################

#resampling for time series
sales_a = train[train.Store == 2]['Sales']
sales_a.resample('W').sum().plot(color = c) #weekly

#trends
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition_a = seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c)

#PACF and ACF
plot_acf(sales_a, lags = 50, ax = plt.gca(), color = c)
plot_pacf(sales_a, lags = 50, ax = plt.gca(), color = c)

# convert python dataframe to datetime/ datetime to month
data['month'] = data["Date"].dt.month

####month/year from date
train = pd.read_csv(".....train.csv", parse_dates = True, low_memory = False, index_col = 'Date')
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear
#####


#boxplot
train.boxplot('life', 'Region', rot=60)
####
plt.clf()
plt.figure(figsize=(13,9))
sn.boxplot(data = complete[continuous_vars])
#######

# sns distplot, combination of histogram for continuous variables and smoothing curve
sn.distplot(a = (complete[cont_vars[i]]), kde = True, color = 'blue')

#one hot encoding for string categorical values . Use this for pandas
df_region = pd.get_dummies(train)
df_region1 = pd.get_dummies(train, drop_first=True)

# encoding into 3 categories: get dummies
pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
combined.drop('Pclass',axis=1,inplace=True)
combined = pd.concat([combined,pclass_dummies],axis=1)


#use this when the categorical values are numerical. Use this for numpy
sklearn.preprocessing.OneHotEncoder()

########crosstabfor categorical variables, element values must be numberic: if not convert it using label encoder
pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
########

# unique categorical labels
df_train['Embarked'].unique()

# histogram of continuous variables.
survived_df['AgeFill'].hist(bins=int(max_age / bin_size), range=(1, max_age))
pd.value_counts(df_attr.name)

# histogram of categorical variables
plt.hist(survived_df['Class'])

# imputation groupby
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()

# sort
result = result.sort_values(['Id'], ascending=[True])

# convert to int32
test['StateHoliday'] = test['StateHoliday'].astype('int32')

# search term in text and keep only if text contains term
df[df['Text'].str.contains('S.Korea')]

########## Normalize words
replacements = {
    'Col_Name': {
        r'\'s': '', 
        'Indian': 'India', 
        '\'': '', 
    }
}
df.replace(replacements, regex=True, inplace=True)
#################

# Play audio
from playsound import playsound


# time it
import timeit
start = timeit.default_timer()
stop = timeit.default_timer()
print (stop - start )

# dataframe string to lower case
df['x'].str.lower()

# count of regex chars/ pattern
df=train["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

############### send notification on android
from urllib.parse import urlencode
from urllib.request import Request, urlopen

url = 'https://www.pushsafer.com/api' # Set destination URL here
post_fields = {                       # Set POST fields here
	"t" : "title",
	"m" : "hello",
	"d" : "a",
	"u" : url,
	"k" : "FRp5YHumcMWMttF4L5g0"
	}

request = Request(url, urlencode(post_fields).encode())
json = urlopen(request).read().decode()
print(json)
###############

# Count frequency of items in list
from collections import Counter
wcount = Counter()
wcount.update(LIST) # wcount is similar to dictionary

# read each line into list
with open("C:\\Users\\Sanket Keni\\Downloads\\english") as f:
    content = f.readlines()
content = [x.strip() for x in content] 














Use PCA for dense matrix and SVD for sparse matrix
Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
Continous: Age, Fare. Discrete: SibSp, Parch.























