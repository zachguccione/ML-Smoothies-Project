import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# load in df
df = pd.read_csv('data/input_features.csv')

########################
# Gender Based Scaling #
########################
scalers = {}

# Gender based columns to scale
columns_to_scale = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI', 'Fat_Percentage']

def gender_based_scaling(df):
    # Ensure `Gender` exists in the DataFrame
    if 'Gender' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Gender' column.")
    
    # Check the gender value in the single-row DataFrame
    if df.iloc[0]['Gender'] == 'Male':
        with open('models/Male_scaler.pkl', 'rb') as f:
            male_scaler = pickle.load(f)
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = male_scaler.transform(df[columns_to_scale])
    elif df.iloc[0]['Gender'] == 'Female':
        with open('models/Female_scaler.pkl', 'rb') as f:
            female_scaler = pickle.load(f)
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = female_scaler.transform(df[columns_to_scale])
    else:
        raise ValueError("Gender must be either 'Male' or 'Female'.")
    
    return df_scaled

df = gender_based_scaling(df)

#################
# Other Scaling #
#################
numerical_columns = ['Age', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)']

def other_numerical_scaling(df):
    with open('models/numerical_scaler.pkl', 'rb') as f:
        numerical_scaler = pickle.load(f)
    df[numerical_columns] = numerical_scaler.transform(df[numerical_columns])
    return df

df = other_numerical_scaling(df)

###################
# One Hot Encoder #
###################
one_hot_columns = ['Gender', 'Workout_Type']

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(df[one_hot_columns])

encoded_df = pd.DataFrame(
    encoded_data, 
    columns=encoder.get_feature_names_out(one_hot_columns),
    index=df.index  
)

# Drop the original columns and concatenate the encoded columns
df = pd.concat([df.drop(columns=one_hot_columns), encoded_df], axis=1)

###################
# Input Massaging #
###################
kmeans_model_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                         'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
                         'Water_Intake (liters)', 'Workout_Frequency (days/week)',
                         'Experience_Level', 'BMI', 'Workout_Type_Cardio', 'Workout_Type_HIIT',
                         'Workout_Type_Strength', 'Gender_Male']

# creating empty columns to match the kmeans model
def create_empty_columns(df):
    for col in kmeans_model_columns:
        if col not in df.columns:
            df[col] = 0
    return df

create_empty_columns(df)

# aligning columns correctly
df = df[kmeans_model_columns]

##############################
# Loading into K-means model #
##############################
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Predict the cluster for the input data
cluster = kmeans_model.predict(df)

# Save the cluster to a file
print(cluster)