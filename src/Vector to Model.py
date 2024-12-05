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

#######################################
# Returning Smoothie Based on Cluster #
#######################################
# Cluster 0
if cluster == 0:
    print("Cluster 0: Cardio-Focused")
    print("")
    print("Building the Smoothie:")
    print("Cinnamon for Anti-inflammatory")
    print("Matcha for Energy")
    print("Berries for Antioxidants/inflammation reduction")
    print("Whey Protein for protein")
    print("Chia seeds for omega-3s and fiber")
# Cluster 1
elif cluster == 1:
    print("Cluster 1: Endurance Booster")
    print("")
    print("Building the Smoothie:")
    print("Coconut water for hydration")
    print("Bananas for sugar and potassium")
    print("Almond butter for sustain energy")
    print("Greek yogurt for protein")
# Cluster 2
elif cluster == 2:
    print("Cluster 2: Strength and High Intensity")
    print("")
    print("Building the Smoothie:")
    print("Whey protein for muscle")
    print("Dairy milk for extra protein")
    print("Almond butter for calories")
    print("Creatine to increase muscle mass")
    print("Rolled oats to provide energy")
# Cluster 3
elif cluster == 3:
    print("Cluster 3: Weight Loss Beginners")
    print("")
    print("Building the Smoothie:")
    print("Almond milk for a low-calorie base")
    print("Lemon and Cucumber for a hydrating effect")
    print("Spinach for micro-nutrients")
    print("Ginger to boost digestion and metabolism")
    print("Watermelon to add sweetness")
else:
    print("Cluster 4: Balance Energy for Beginners")
    print("")
    print("​Building the Smoothie:")
    print("Apple for fiber and sweetness")
    print("Coconut water for hydration")
    print("Greek yogurt for protein")
    print("Chia seeds for nutrition boost and satiation")