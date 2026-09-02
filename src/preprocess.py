import pandas as pd

df = pd.read_csv('data/phishing_email.csv')
print(df.shape)
print(df['label'].value_counts())
print(df.isnull().sum())
print(df.head())
print(df[df['label']==1]['text_combined'].iloc[0][:300])
print(df[df['label']==1]['text_combined'].iloc[5][:300])