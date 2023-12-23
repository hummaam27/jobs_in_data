# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.681243Z","iopub.execute_input":"2023-12-21T16:46:52.681551Z","iopub.status.idle":"2023-12-21T16:46:52.687978Z","shell.execute_reply.started":"2023-12-21T16:46:52.681526Z","shell.execute_reply":"2023-12-21T16:46:52.687218Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
        
pd.options.mode.chained_assignment = None

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # 1.Get an understanding of the dataset
# 1. Inspect the data
# 1. Check for null values
# 1. Check data types
# 1. Get some summary statistics

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.694014Z","iopub.execute_input":"2023-12-21T16:46:52.694243Z","iopub.status.idle":"2023-12-21T16:46:52.711853Z","shell.execute_reply.started":"2023-12-21T16:46:52.694222Z","shell.execute_reply":"2023-12-21T16:46:52.711013Z"}}
#import the data
df = pd.read_csv("C:\\Users\\humma\\Downloads\\salaries.csv")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.714145Z","iopub.execute_input":"2023-12-21T16:46:52.714412Z","iopub.status.idle":"2023-12-21T16:46:52.728808Z","shell.execute_reply.started":"2023-12-21T16:46:52.714389Z","shell.execute_reply":"2023-12-21T16:46:52.727965Z"}}
#1.1 - quick view of the data
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.729566Z","iopub.execute_input":"2023-12-21T16:46:52.729796Z","iopub.status.idle":"2023-12-21T16:46:52.741460Z","shell.execute_reply.started":"2023-12-21T16:46:52.729776Z","shell.execute_reply":"2023-12-21T16:46:52.740469Z"}}
#1.2 - Check the data types and see if are null values
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:52.742366Z","iopub.execute_input":"2023-12-21T16:46:52.742571Z","iopub.status.idle":"2023-12-21T16:46:52.754401Z","shell.execute_reply.started":"2023-12-21T16:46:52.742551Z","shell.execute_reply":"2023-12-21T16:46:52.753770Z"}}
df['company_location'].unique()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.756453Z","iopub.execute_input":"2023-12-21T16:46:52.756867Z","iopub.status.idle":"2023-12-21T16:46:52.766724Z","shell.execute_reply.started":"2023-12-21T16:46:52.756844Z","shell.execute_reply":"2023-12-21T16:46:52.766169Z"}}
#1.2 - Check for any null values
df.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.767815Z","iopub.execute_input":"2023-12-21T16:46:52.768252Z","iopub.status.idle":"2023-12-21T16:46:52.777734Z","shell.execute_reply.started":"2023-12-21T16:46:52.768229Z","shell.execute_reply":"2023-12-21T16:46:52.776969Z"}}
#1.3 - Check data types 
df.dtypes

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.779033Z","iopub.execute_input":"2023-12-21T16:46:52.779320Z","iopub.status.idle":"2023-12-21T16:46:52.798000Z","shell.execute_reply.started":"2023-12-21T16:46:52.779300Z","shell.execute_reply":"2023-12-21T16:46:52.797039Z"}}
#1.4 - Some summary statistics fot the 'salary' and 'salary_in_usd columns'
df[['salary','salary_in_usd']].describe()

# %% [markdown]
# # 2. Data Filtering and Cleaning
# 
# 1. Replace values of vague columns and rename columns
# 1. Check for outliers and remove if necessary.
# 1. use a function to categorize the job_title values into a new column
# 1. create some vizualisaions to undertsand the data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.799245Z","iopub.execute_input":"2023-12-21T16:46:52.799491Z","iopub.status.idle":"2023-12-21T16:46:52.806641Z","shell.execute_reply.started":"2023-12-21T16:46:52.799471Z","shell.execute_reply":"2023-12-21T16:46:52.806022Z"}}
#2.1 - Make 'experince_level column more readable
df['experience_level'].replace(['SE', 'EX', 'EN', 'MI'],['Senior', 'Executive','Entry-level', 'Mid-level'], inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:48:51.112767Z","iopub.execute_input":"2023-12-21T16:48:51.113117Z","iopub.status.idle":"2023-12-21T16:48:51.130524Z","shell.execute_reply.started":"2023-12-21T16:48:51.113092Z","shell.execute_reply":"2023-12-21T16:48:51.129783Z"}}
df

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:53:53.336648Z","iopub.execute_input":"2023-12-21T16:53:53.336955Z","iopub.status.idle":"2023-12-21T16:53:53.345044Z","shell.execute_reply.started":"2023-12-21T16:53:53.336934Z","shell.execute_reply":"2023-12-21T16:53:53.344213Z"}}
#2.1 - replace country initials
def replace_country_initials(df):
    # Updated mapping of country initials to full names
    country_mapping = {
        'DE': 'Germany', 'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
        'ES': 'Spain', 'IE': 'Ireland', 'ZA': 'South Africa', 'PL': 'Poland',
        'FR': 'France', 'NL': 'Netherlands', 'LU': 'Luxembourg', 'LT': 'Lithuania',
        'PT': 'Portugal', 'GI': 'Gibraltar', 'AU': 'Australia', 'CO': 'Colombia',
        'UA': 'Ukraine', 'SI': 'Slovenia', 'IN': 'India', 'RO': 'Romania',
        'GR': 'Greece', 'LV': 'Latvia', 'MU': 'Mauritius', 'RU': 'Russia',
        'IT': 'Italy', 'KR': 'South Korea', 'EE': 'Estonia', 'CZ': 'Czech Republic',
        'CH': 'Switzerland', 'BR': 'Brazil', 'QA': 'Qatar', 'KE': 'Kenya',
        'DK': 'Denmark', 'GH': 'Ghana', 'SE': 'Sweden', 'PH': 'Philippines',
        'TR': 'Turkey', 'AD': 'Andorra', 'EC': 'Ecuador', 'MX': 'Mexico',
        'IL': 'Israel', 'NG': 'Nigeria', 'SA': 'Saudi Arabia', 'NO': 'Norway',
        'AR': 'Argentina', 'JP': 'Japan', 'HK': 'Hong Kong', 'CF': 'Central African Republic',
        'FI': 'Finland', 'SG': 'Singapore', 'TH': 'Thailand', 'HR': 'Croatia',
        'AM': 'Armenia', 'BA': 'Bosnia and Herzegovina', 'PK': 'Pakistan',
        'IR': 'Iran', 'BS': 'Bahamas', 'HU': 'Hungary', 'AT': 'Austria',
        'PR': 'Puerto Rico', 'AS': 'American Samoa', 'BE': 'Belgium', 'ID': 'Indonesia',
        'EG': 'Egypt', 'AE': 'United Arab Emirates', 'MY': 'Malaysia',
        'HN': 'Honduras', 'DZ': 'Algeria', 'IQ': 'Iraq', 'CN': 'China',
        'NZ': 'New Zealand', 'CL': 'Chile', 'MD': 'Moldova', 'MT': 'Malta'
    }
    
    # Replace the initials with full country names
    df['company_location'] = df['company_location'].map(country_mapping).fillna(df['company_location'])
    return df

df = replace_country_initials(df)
df


# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:53:53.729521Z","iopub.execute_input":"2023-12-21T16:53:53.729840Z","iopub.status.idle":"2023-12-21T16:53:53.736267Z","shell.execute_reply.started":"2023-12-21T16:53:53.729819Z","shell.execute_reply":"2023-12-21T16:53:53.735484Z"}}
# 2.1 - check if function worked
df['company_location']

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:58:46.906541Z","iopub.execute_input":"2023-12-21T16:58:46.906864Z","iopub.status.idle":"2023-12-21T16:58:46.924045Z","shell.execute_reply.started":"2023-12-21T16:58:46.906840Z","shell.execute_reply":"2023-12-21T16:58:46.923463Z"}}
df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.834714Z","iopub.execute_input":"2023-12-21T16:46:52.834996Z","iopub.status.idle":"2023-12-21T16:46:52.844608Z","shell.execute_reply.started":"2023-12-21T16:46:52.834976Z","shell.execute_reply":"2023-12-21T16:46:52.844003Z"}}
#2.1 - Make 'employment_type' column more readable
df['employment_type'].replace(['PT', 'FT', 'CT', 'FL'],['Part-time', 'Full-time', 'Contract', 'Freelance'], inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.845565Z","iopub.execute_input":"2023-12-21T16:46:52.845924Z","iopub.status.idle":"2023-12-21T16:46:52.856098Z","shell.execute_reply.started":"2023-12-21T16:46:52.845902Z","shell.execute_reply":"2023-12-21T16:46:52.855349Z"}}
#2.1 - Check values of the `remote_ratio` columns
df['remote_ratio'].unique()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.857138Z","iopub.execute_input":"2023-12-21T16:46:52.857352Z","iopub.status.idle":"2023-12-21T16:46:52.866369Z","shell.execute_reply.started":"2023-12-21T16:46:52.857332Z","shell.execute_reply":"2023-12-21T16:46:52.865598Z"}}
#2.1 - Its values aren't very descriptive. we'll replace them to make them more readable.
df.rename(columns={'remote_ratio': 'work_setting'}, inplace=True)
df['work_setting'].replace([100, 50, 0],['Remote', 'Hybrid', 'In-person'], inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:52.869706Z","iopub.execute_input":"2023-12-21T16:46:52.870339Z","iopub.status.idle":"2023-12-21T16:46:53.007357Z","shell.execute_reply.started":"2023-12-21T16:46:52.870317Z","shell.execute_reply":"2023-12-21T16:46:53.006782Z"}}
#2.2 - Use a boxplot to look for outliers
plt.figure(figsize=(12, 1))
sns.boxplot(x=df['salary_in_usd'])
plt.title('Boxplot of Salary in USD');

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.008243Z","iopub.execute_input":"2023-12-21T16:46:53.009058Z","iopub.status.idle":"2023-12-21T16:46:53.016201Z","shell.execute_reply.started":"2023-12-21T16:46:53.009031Z","shell.execute_reply":"2023-12-21T16:46:53.015583Z"}}
#2.2 - Check outliers
df['salary_in_usd'].sort_values(ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.017006Z","iopub.execute_input":"2023-12-21T16:46:53.017976Z","iopub.status.idle":"2023-12-21T16:46:53.025083Z","shell.execute_reply.started":"2023-12-21T16:46:53.017951Z","shell.execute_reply":"2023-12-21T16:46:53.024438Z"}}
#2.2 - Filter out the outliers
df = df[df['salary'] <= 450000]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.025945Z","iopub.execute_input":"2023-12-21T16:46:53.026697Z","iopub.status.idle":"2023-12-21T16:46:53.036775Z","shell.execute_reply.started":"2023-12-21T16:46:53.026672Z","shell.execute_reply":"2023-12-21T16:46:53.035947Z"}}
#2.2 - Check the filter
df['salary'].sort_values(ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.037952Z","iopub.execute_input":"2023-12-21T16:46:53.038265Z","iopub.status.idle":"2023-12-21T16:46:53.046840Z","shell.execute_reply.started":"2023-12-21T16:46:53.038238Z","shell.execute_reply":"2023-12-21T16:46:53.045958Z"}}
# 2.3 - check how many unique job titles we have in the dataset
df['job_title'].unique()

# %% [markdown]
# Currently we have 105 unqiue job titles. Though this is good for detail, it is very difficult to vizualize and dril down. We need to categorize these to make them better suited for analysis. We can create a function that will parse the data and add a new column with the job titles categorized.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.047964Z","iopub.execute_input":"2023-12-21T16:46:53.048806Z","iopub.status.idle":"2023-12-21T16:46:53.115965Z","shell.execute_reply.started":"2023-12-21T16:46:53.048778Z","shell.execute_reply":"2023-12-21T16:46:53.115172Z"}}
# 2.3 - create a function that will categorize the job titles into a new column
def group_job_titles(job_title):
    # Define keywords for each specific category
    core_data_analysis_keywords = ['Data Analyst', 'Financial', 'Business Data', 'Compliance', 'Product Data']
    advanced_data_science_keywords = ['Data Scientist', 'Research Scientist', 'Research Analyst', 'Applied Scientist', 'Decision Scientist']
    ml_development_keywords = ['Machine Learning Researcher', 'Machine Learning Engineer', 'Deep Learning', 'NLP', 'Computer Vision Engineer', 'ML Scientist', 'Applied ML']
    ai_specialization_keywords = ['AI Engineer', 'AI Developer', 'AI Architect', 'AI Research', 'AI Scientist']
    ml_ops_keywords = ['MLOps', 'ML Infrastructure', 'ML Operations', 'ML Software', 'ML Manager', 'ML Modeler']
    data_engineering_keywords = ['Data Engineer', 'Data Integration', 'ETL', 'Big Data', 'Cloud Data Engineer', 'Data Infrastructure', 'Software Data Engineer']
    data_architecture_keywords = ['Data Architect', 'Data Modeler', 'Data Strategist', 'Data Strategy', 'Cloud Data Architect', 'AWS Data Architect']
    bi_visualization_keywords = ['BI Developer', 'Business Intelligence Engineer', 'BI Analyst', 'Data Visualization', 'BI Data Analyst']
    data_quality_ops_keywords = ['Data Quality', 'Data Operations']
    data_product_mgmt_keywords = ['Data Product Manager', 'Data Manager', 'Data Lead', 'Data Science Manager', 'Data Analytics Manager', 'Manager Data Management', 'Data Developer', 'Data Science Practitioner']
    analytics_engineering_keywords = ['Analytics Engineer', 'Data Analytics Specialist', 'Data Analytics Consultant', 'Data Analytics Engineer', 'Data Analytics Lead']
    leadership_executive_keywords = ['Data Science Director', 'Head of Data', 'Director of Data Science', 'Head of ML', 'Head of Data Science', 'Managing Director Data Science', 'Lead', 'Principal']
    senior_principal_roles_keywords = ['Staff', 'Principal']
    cloud_database_keywords = ['Cloud Database Engineer', 'BI Data Engineer']
    data_management_analysis_keywords = ['Data Management Analyst', 'Data Specialist', 'Business Intelligence Manager', 'Business Intelligence Developer', 'Business Intelligence Specialist']
    
       # Check and assign categories based on keywords
    if any(keyword in job_title for keyword in core_data_analysis_keywords):
        return 'Core Data Analysis'
    elif any(keyword in job_title for keyword in advanced_data_science_keywords):
        return 'Advanced Data Science and Research'
    elif any(keyword in job_title for keyword in ml_development_keywords):
        return 'ML Development and Research'
    elif any(keyword in job_title for keyword in ai_specialization_keywords):
        return 'AI Specialization'
    elif any(keyword in job_title for keyword in ml_ops_keywords):
        return 'ML Operations and Infrastructure'
    elif any(keyword in job_title for keyword in data_engineering_keywords):
        return 'Data Engineering and Integration'
    elif any(keyword in job_title for keyword in data_architecture_keywords):
        return 'Data Architecture, Strategy, and Modeling'
    elif any(keyword in job_title for keyword in bi_visualization_keywords):
        return 'BI and Visualization'
    elif any(keyword in job_title for keyword in data_quality_ops_keywords):
        return 'Data Quality and Operations Management'
    elif any(keyword in job_title for keyword in data_product_mgmt_keywords):
        return 'Data Product and Project Management'
    elif any(keyword in job_title for keyword in analytics_engineering_keywords):
        return 'Analytics Engineering and Consulting'
    elif any(keyword in job_title for keyword in leadership_executive_keywords):
        return 'Leadership and Executive Roles'
    elif any(keyword in job_title for keyword in senior_principal_roles_keywords):
        return 'Senior and Principal Roles'
    elif any(keyword in job_title for keyword in cloud_database_keywords):
        return 'Cloud and Database Specialization'
    elif any(keyword in job_title for keyword in data_management_analysis_keywords):
        return 'Data Management and Analysis'
    else:
        return 'Other'

# Applying the function to the job_title column
df['job_category'] = df['job_title'].apply(group_job_titles)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.117145Z","iopub.execute_input":"2023-12-21T16:46:53.117381Z","iopub.status.idle":"2023-12-21T16:46:53.124352Z","shell.execute_reply.started":"2023-12-21T16:46:53.117361Z","shell.execute_reply":"2023-12-21T16:46:53.123497Z"}}
# 2.3 - check the function
df['job_category'].value_counts()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-21T16:46:53.125412Z","iopub.execute_input":"2023-12-21T16:46:53.125881Z","iopub.status.idle":"2023-12-21T16:46:53.142532Z","shell.execute_reply.started":"2023-12-21T16:46:53.125859Z","shell.execute_reply":"2023-12-21T16:46:53.141543Z"}}
#2.3 - check for null values
df.info()

# %% [markdown]
# We were able to categorize 94% of the values. The remaining 6% are categorized under 'other'. Depending on the analysis, these values can be removed or left. For this excercise, I will leave them in.**

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:53.143951Z","iopub.execute_input":"2023-12-21T16:46:53.144708Z","iopub.status.idle":"2023-12-21T16:46:53.426087Z","shell.execute_reply.started":"2023-12-21T16:46:53.144679Z","shell.execute_reply":"2023-12-21T16:46:53.425289Z"}}
# 2.4 - Distribution of Salaries
plt.figure(figsize=(10, 6))
sns.histplot(df['salary'])
plt.title('Distribution of Salaries')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:53.427130Z","iopub.execute_input":"2023-12-21T16:46:53.427575Z","iopub.status.idle":"2023-12-21T16:46:53.819654Z","shell.execute_reply.started":"2023-12-21T16:46:53.427553Z","shell.execute_reply":"2023-12-21T16:46:53.818800Z"}}
# 2.4 - Boxplot of salaries by job category
plt.figure(figsize=(15, 7))
sns.boxplot(x='salary', y='job_category', data=df)
plt.title('Salary by Job Category')
plt.xlabel('Salary (USD)')
plt.ylabel('Job Category')
plt.xticks(rotation=45)
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:53.820684Z","iopub.execute_input":"2023-12-21T16:46:53.821137Z","iopub.status.idle":"2023-12-21T16:46:54.025560Z","shell.execute_reply.started":"2023-12-21T16:46:53.821112Z","shell.execute_reply":"2023-12-21T16:46:54.024721Z"}}
# 2.4 - Barplot of average salary by experience

avg_salary_by_experience = df.groupby('experience_level')['salary'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='experience_level', y='salary', data=avg_salary_by_experience)
plt.title('Average Salary by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (USD)')
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:54.026786Z","iopub.execute_input":"2023-12-21T16:46:54.027008Z","iopub.status.idle":"2023-12-21T16:46:54.147604Z","shell.execute_reply.started":"2023-12-21T16:46:54.026987Z","shell.execute_reply":"2023-12-21T16:46:54.146780Z"}}
# 2.4 - Work Settings Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['work_setting'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Work Settings')
plt.ylabel('')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-21T16:46:54.148891Z","iopub.execute_input":"2023-12-21T16:46:54.149137Z","iopub.status.idle":"2023-12-21T16:46:54.201470Z","shell.execute_reply.started":"2023-12-21T16:46:54.149113Z","shell.execute_reply":"2023-12-21T16:46:54.200797Z"}}
df.to_excel("C:\\Users\\humma\\Downloads\\data_jobs_complete.xlsx", index = False)

# %% [code]
