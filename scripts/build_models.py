import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_users_2.csv')

# Drop to prevent leakage of information from the target variable
# An existing value for date_first_booking would indicate that the user has already booked, which is what we are trying to predict
df = df.drop(["date_first_booking"])



