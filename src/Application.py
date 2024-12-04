import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import model
model = joblib.load('src/kmeans_model.pkl')

# features
feature_names = [
       'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
       'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
       'Workout_Type', 'Fat_Percentage', 'Water_Intake (liters)',
       'Workout_Frequency (days/week)', 'Experience_Level', 'BMI'
]

# Function to handle prediction
def predict_cluster():
    try:
        # Collect input values
        age = float(entry_age.get())
        weight = float(entry_weight.get())
        gender = gender_var.get()
        height = float(entry_height.get())
        resting_bpm = float(entry_resting_bpm.get())
        max_bpm = float(entry_max_bpm.get())
        avg_bpm = float(entry_avg_bpm.get())
        session_duration = float(entry_session_duration.get())
        calories_burned = float(entry_calories_burned.get())
        fat_percentage = float(entry_fat_percentage.get())
        water_intake = float(entry_water_intake.get())
        workout_frequency = int(workout_frequency_var.get())
        workout_type = workout_type_var.get()
        experience_level = experience_level_var.get()
        bmi = weight / height ** 2

        # Example feature vector
        features = np.array([
        age, gender, weight, height, max_bpm, avg_bpm,
        resting_bpm, session_duration, calories_burned,
        workout_type, fat_percentage, water_intake,
        workout_frequency, experience_level, bmi]).reshape(1, -1)
        
        # Saving the dataframe of our features
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scaling our features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df[['Age', 'Height (m)', 'Weight (kg)', 'Calories_Burned', 'BMI']])

        print("Features:", features_df)
        # Placeholder for predicted cluster
        cluster = model.predict(features_df)
        messagebox.showinfo("Result", f"The feature vector belongs to Cluster: {cluster}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values in all fields.")

# Create the main application window
app = ttk.Window(themename="litera")  # Choose a theme: "litera", "darkly", etc.
app.title("Smoothie Predictor")
app.geometry("750x825")

# Define column labels
ttk.Label(app, text="Personal Information", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=20, pady=10, sticky="w")
ttk.Label(app, text="Workout Information", font=("Arial", 14, "bold")).grid(row=0, column=1, padx=20, pady=10, sticky="w")

# Personal Information Section
gender_var = ttk.IntVar(value=1)
ttk.Label(app, text="Gender:", font=("Arial", 12)).grid(row=1, column=0, sticky="w", padx=20, pady=5)
ttk.Radiobutton(app, text="Male", variable=gender_var, value=1, bootstyle="info").grid(row=2, column=0, sticky="w", padx=40)
ttk.Radiobutton(app, text="Female", variable=gender_var, value=0, bootstyle="info").grid(row=3, column=0, sticky="w", padx=40)

ttk.Label(app, text="Age:", font=("Arial", 12)).grid(row=4, column=0, sticky="w", padx=20, pady=5)
entry_age = ttk.Entry(app, width=30)
entry_age.grid(row=5, column=0, padx=20, sticky="w")

ttk.Label(app, text="Weight (kg):", font=("Arial", 12)).grid(row=6, column=0, sticky="w", padx=20, pady=5)
entry_weight = ttk.Entry(app, width=30)
entry_weight.grid(row=7, column=0, padx=20, sticky="w")

ttk.Label(app, text="Height (m):", font=("Arial", 12)).grid(row=8, column=0, sticky="w", padx=20, pady=5)
entry_height = ttk.Entry(app, width=30)
entry_height.grid(row=9, column=0, padx=20, sticky="w")

ttk.Label(app, text="Resting BPM:", font=("Arial", 12)).grid(row=10, column=0, sticky="w", padx=20, pady=5)
entry_resting_bpm = ttk.Entry(app, width=30)
entry_resting_bpm.grid(row=11, column=0, padx=20, sticky="w")

workout_type_var = ttk.IntVar(value=1)
ttk.Label(app, text="Workout Type:", font=("Arial", 12)).grid(row=1, column=1, sticky="w", padx=20, pady=5)
ttk.Radiobutton(app, text="HIIT", variable=workout_type_var, value=0, bootstyle="success").grid(row=2, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Yoga", variable=workout_type_var, value=1, bootstyle="success").grid(row=3, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Cardio", variable=workout_type_var, value=2, bootstyle="success").grid(row=4, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Strength", variable=workout_type_var, value=3, bootstyle="success").grid(row=5, column=1, sticky="w", padx=40, pady=5)

ttk.Label(app, text="Max BPM:", font=("Arial", 12)).grid(row=6, column=1, sticky="w", padx=20, pady=5)
entry_max_bpm = ttk.Entry(app, width=30)
entry_max_bpm.grid(row=7, column=1, padx=20, sticky="w")

ttk.Label(app, text="Avg BPM:", font=("Arial", 12)).grid(row=8, column=1, sticky="w", padx=20, pady=5)
entry_avg_bpm = ttk.Entry(app, width=30)
entry_avg_bpm.grid(row=9, column=1, padx=20, sticky="w")

ttk.Label(app, text="Session Duration (hours):", font=("Arial", 12)).grid(row=10, column=1, sticky="w", padx=20, pady=5)
entry_session_duration = ttk.Entry(app, width=30)
entry_session_duration.grid(row=11, column=1, padx=20, sticky="w")

ttk.Label(app, text="Calories Burned:", font=("Arial", 12)).grid(row=12, column=1, sticky="w", padx=20, pady=5)
entry_calories_burned = ttk.Entry(app, width=30)
entry_calories_burned.grid(row=13, column=1, padx=20, sticky="w")

# Additional Inputs
ttk.Label(app, text="Fat Percentage:", font=("Arial", 12)).grid(row=12, column=0, sticky="w", padx=20, pady=5)
entry_fat_percentage = ttk.Entry(app, width=30)
entry_fat_percentage.grid(row=13, column=0, padx=20, sticky="w")

ttk.Label(app, text="Water Intake (Liters):", font=("Arial", 12)).grid(row=14, column=1, sticky="w", padx=20, pady=5)
entry_water_intake = ttk.Entry(app, width=30)
entry_water_intake.grid(row=15, column=1, padx=20, sticky="w")

ttk.Label(app, text="Workout Frequency (Days/week):", font=("Arial", 12)).grid(row=16, column=1, sticky="w", padx=20, pady=5)
workout_frequency_var = ttk.StringVar(value="1")  # Default value
workout_frequency_options = ["1", "1", "2", "3", "4", "5"]
dropdown_workout_frequency = ttk.OptionMenu(app, workout_frequency_var, *workout_frequency_options)
dropdown_workout_frequency.grid(row=17, column=1, padx=20, sticky="w")

# Experience Level
experience_level_var = ttk.IntVar(value=1)
ttk.Label(app, text="Experience Level:", font=("Arial", 12)).grid(row=14, column=0, sticky="w", padx=20, pady=5)
ttk.Radiobutton(app, text="Beginner", variable=experience_level_var, value=1, bootstyle="warning").grid(row=15, column=0, sticky="w", padx=40)
ttk.Radiobutton(app, text="Intermediate", variable=experience_level_var, value=2, bootstyle="warning").grid(row=16, column=0, sticky="w", padx=40)
ttk.Radiobutton(app, text="Advanced", variable=experience_level_var, value=3, bootstyle="warning").grid(row=17, column=0, sticky="w", padx=40)

# Predict Button
ttk.Button(
    app, 
    text="Predict", 
    command=predict_cluster, 
    bootstyle="primary"
).grid(row=22, column=0, columnspan=2, pady=30, sticky="w", padx=300)

# Run the application
app.mainloop()