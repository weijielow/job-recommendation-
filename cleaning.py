# For reading data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data = pd.read_csv('naukri_com-job_sample.csv')

# Display first few rows
data.head()
# Display dataset information
print("Dataset Information:")
print(data.info())
# Detect missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)


# Create a pairplot to visualize relationships between numeric columns
sns.pairplot(data[['education', 'experience', 'numberofpositions', 'payrate', 'industry']], hue='industry')
plt.title('Pairplot of Education, Experience, Number of Positions, Pay Rate, and Industry')
plt.show()



# Create a bar chart to visualize the distribution of industries
plt.figure(figsize=(10, 6))
sns.countplot(data['industry'], order=data['industry'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Industries')
plt.show()

# Create a bar chart to visualize the distribution of job titles
plt.figure(figsize=(12, 6))
sns.countplot(data['jobtitle'], order=data['jobtitle'].value_counts().index[:10])
plt.xticks(rotation=90)
plt.title('Top 10 Job Titles')
plt.show()

# Create a bar chart to visualize the distribution of skills
skills_df = data['skills'].str.split(',').explode().str.strip()
plt.figure(figsize=(12, 6))
sns.countplot(skills_df, order=skills_df.value_counts().index[:10])
plt.xticks(rotation=90)
plt.title('Top 10 Skills')
plt.show()
#Drop unnecessarily data columns
data.drop(['numberofpositions','site_name','jobid','uniq_id'],axis=1,inplace=True)
#Impute null data for education and skills
from sklearn.impute import SimpleImputer
to_fill = ['education', 'skills']
for col in to_fill:
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_values = imputer.fit_transform(data[[col]])
    data[col] = imputed_values[:, 0]

experience_lower = []
experience_upper = []
invalid = []
for idx, row in data.iterrows():
    try:
        text = re.sub('yrs', '', row['experience'])
        splits = text.split('-')
        experience_lower.append(int(splits[0]))
        experience_upper.append(int(splits[1]))
    except:
        invalid.append(row['experience'])

data = data[~data['experience'].isin(invalid)]
data['experience_lower'] = data['experience'].apply(lambda x: int(x.split('-')[0]))
data['experience_upper'] = data['experience'].apply(lambda x: int(re.sub('yrs','', x.split('-')[1])))
data.drop(['experience'], axis=1, inplace=True)

#Drop the unneeded last 5 string value of each data in postdate column
data['postdate'] = data['postdate'].astype(str).apply(lambda x: x[:-5])

data['job_age']=pd.to_datetime('today') - pd.to_datetime(data['postdate'])
data['job_age'] = data['job_age'].dt.days

replacements = {
   'joblocation_address': {
      r'(Bengaluru/Bangalore)': 'Bangalore',
      r'Bengaluru': 'Bangalore',
      r'Hyderabad / Secunderabad': 'Hyderabad',
      r'Mumbai , Mumbai': 'Mumbai',
      r'Noida': 'NCR',
      r'Delhi': 'NCR',
      r'Gurgaon': 'NCR',
      r'Delhi/NCR(National Capital Region)': 'NCR',
      r'Delhi , Delhi': 'NCR',
      r'Noida , Noida/Greater Noida': 'NCR',
      r'Ghaziabad': 'NCR',
      r'Delhi/NCR(National Capital Region) , Gurgaon': 'NCR',
      r'NCR , NCR': 'NCR',
      r'NCR/NCR(National Capital Region)': 'NCR',
      r'NCR , NCR/Greater NCR': 'NCR',
      r'NCR/NCR(National Capital Region) , NCR': 'NCR',
      r'NCR , NCR/NCR(National Capital Region)': 'NCR',
      r'Bangalore , Bangalore / Bangalore': 'Bangalore',
      r'Bangalore , karnataka': 'Bangalore',
      r'NCR/NCR(National Capital Region)': 'NCR',
      r'NCR/Greater NCR': 'NCR',
      r'NCR/NCR(National Capital Region) , NCR': 'NCR'

   }
}

data.replace(replacements, regex=True, inplace=True)

data['industry'] = data['industry'].astype(str).apply(lambda x: x.split('/')[0])
data['industry'] = data['industry'].str.strip()
data['education'] = data['education'].str.split(' ').apply(lambda x: x[1] if len(x) > 1 else x[0])
data['education'] = data['education'].replace(('B.Tech/B.E.','Graduation','Other','-','Not','B.Tech/B.E.,','Postgraduate',
                                               'PG:CA','Diploma,','B.Com,','B.Pharma,','B.A,','BCA,','B.Sc,','MBA/PGDM','B.B.A,',
                                              'PG:Other','Doctorate:Doctorate','Post'),
                                              ('B.Tech','B.Tech','B.Tech','B.Tech','B.Tech','B.Tech','B.Tech',
                                              'CA','Diploma','B.Com','B.Pharma','B.A','BCA','B.Sc','MBA','BBA',
                                              'B.Tech','Doctorate','B.Tech'))


data['skills'] = data['skills'].str.split(" - ").apply(lambda x: x[1] if len(x) > 1 else x[0])

# Removing jobs which have less than 10 postings as they are very rare and affect model performance
majority_industries = data['industry'].value_counts()[data['industry'].value_counts()>=10].index
data = data[data['industry'].isin(majority_industries)]
data.isnull().sum()[data.isnull().sum()>0]

joblocation_imputer = SimpleImputer(strategy='most_frequent')
imputed_values = joblocation_imputer.fit_transform(data[['joblocation_address']])
data['joblocation_address'] = imputed_values[:, 0]
jobage_imputer = SimpleImputer(strategy='mean')
data['job_age'] = jobage_imputer.fit_transform(data[['job_age']])
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

data.to_csv('cleaned_naukri_com-job_sample.csv', index=False)