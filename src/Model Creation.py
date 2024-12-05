import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle

# load in df
df = pd.read_csv('data/gym_members_exercise_tracking.csv')

########################
# Gender Based Scaling #
########################
# Create a dictionary to store scalers
scalers = {}

# Gender based columns to scale
columns_to_scale = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI', 'Fat_Percentage']

# Separate by gender and create a scaler for each
for gender in df['Gender'].unique():
    scaler = StandardScaler()
    scalers[gender] = scaler
    # Fit the scaler on the data for the given gender
    scalers[gender].fit(df[df['Gender'] == gender][columns_to_scale])

    # Save the scaler to a file
    with open("models/"+f'{gender}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Example: Verify loading a scaler and using it
with open('models/Male_scaler.pkl', 'rb') as f:
    male_scaler = pickle.load(f)

# Example: Verify loading a scaler and using it
with open('models/Female_scaler.pkl', 'rb') as f:
    female_scaler = pickle.load(f)

# Transform the male subset using the loaded scaler
male_df = df[df['Gender'] == 'Male']
male_df_scaled = male_df.copy()
male_df_scaled[columns_to_scale] = male_scaler.transform(male_df[columns_to_scale])

female_df = df[df['Gender'] == 'Female']
female_df_scaled = female_df.copy()
female_df_scaled[columns_to_scale] = female_scaler.transform(female_df[columns_to_scale])

# Concatenate both scaled DataFrames
df = pd.concat([male_df_scaled, female_df_scaled]).sort_index()


#################
# Other Scaling #
#################
# Columns to StandardScaler
numerical_columns = ['Age', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)']

# Apply StandardScaler to all numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the scaler to a file
with open('models/numerical_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


###################
# One Hot Encoder #
###################
# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the specified columns
encoded_columns = encoder.fit_transform(df[['Workout_Type', 'Gender']])

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(
    encoded_columns, 
    columns=encoder.get_feature_names_out(['Workout_Type', 'Gender'])
)

# Concatenate the encoded columns back with the original DataFrame
df = pd.concat([df.drop(columns=['Workout_Type', 'Gender']), encoded_df], axis=1)
df.drop(columns=['Gender_Female','Workout_Type_Yoga'], inplace=True)

########################
# Create K-means Model #
########################
# Apply K-Means Clustering with k=5
kmeans = KMeans(n_clusters=5, random_state=42)

clusters = kmeans.fit_predict(df)

# Save the scaler to a file
with open('models/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Access feature names if using a scikit-learn pipeline or preprocessor
if hasattr(kmeans, 'feature_names_in_'):
    print("Input columns expected by the model:", kmeans.feature_names_in_)

print(df)