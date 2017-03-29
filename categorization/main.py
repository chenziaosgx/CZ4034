import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("data.json", orient='records')
training_set , testing_set = train_test_split(df, train_size= 0.7, random_state=42)
print training_set['name'].count()
print testing_set['name'].count()
training_set.to_csv("training.csv", encoding='utf8')
testing_set.to_csv("testing.csv", encoding='utf8')

print "1"