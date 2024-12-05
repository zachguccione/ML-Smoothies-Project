import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import numpy as np
import joblib
import pandas as pd
import csv

# features
feature_names = ['Age' 'Height (m)' 'Max_BPM' 'Avg_BPM' 'Resting_BPM'
 'Session_Duration (hours)' 'Calories_Burned' 'Fat_Percentage'
 'Water_Intake (liters)' 'Workout_Frequency (days/week)'
 'Experience_Level' 'BMI' 'Gender_Male' 'Workout_Type_Cardio'
 'Workout_Type_HIIT' 'Workout_Type_Strength' 'Workout_Type_Yoga']

def save_input_vector_to_dataframe():
    try:
        # Collect input values
        age = float(entry_age.get())
        gender = "Male" if gender_var.get() == 1 else "Female"
        weight = float(entry_weight.get())
        height = float(entry_height.get())
        max_bpm = float(entry_max_bpm.get())
        avg_bpm = float(entry_avg_bpm.get())
        resting_bpm = float(entry_resting_bpm.get())
        session_duration = float(entry_session_duration.get())
        calories_burned = float(entry_calories_burned.get())
        workout_type = workout_type_var.get()  # This is the workout type as a string (e.g., "Strength")
        fat_percentage = float(entry_fat_percentage.get())
        water_intake = float(entry_water_intake.get())
        workout_frequency = int(workout_frequency_var.get())
        experience_level = experience_level_var.get()
        bmi = weight / height ** 2

        # Map workout type to one-hot encoding
        workout_type_map = {
            "Cardio": [1, 0, 0, 0],
            "HIIT": [0, 1, 0, 0],
            "Strength": [0, 0, 1, 0],
            "Yoga": [0, 0, 0, 1],
        }
        if workout_type not in workout_type_map:
            raise ValueError(f"Invalid Workout_Type: {workout_type}")
        workout_type_one_hot = workout_type_map[workout_type]

        # Construct a dictionary to mimic the DataFrame
        input_data = {
            "Age": [age],
            "Gender": [gender],
            "Weight (kg)": [weight],
            "Height (m)": [height],
            "Max_BPM": [max_bpm],
            "Avg_BPM": [avg_bpm],
            "Resting_BPM": [resting_bpm],
            "Session_Duration (hours)": [session_duration],
            "Calories_Burned": [calories_burned],
            "Workout_Type": [workout_type],
            "Fat_Percentage": [fat_percentage],
            "Water_Intake (liters)": [water_intake],
            "Workout_Frequency (days/week)": [workout_frequency],
            "Experience_Level": [experience_level],
            "BMI": [bmi],
        }

        # Create a DataFrame
        input_df = pd.DataFrame(input_data)

        # Save to a CSV file for further processing
        csv_file = "data/input_features.csv"
        input_df.to_csv(csv_file, index=False)

        messagebox.showinfo("Success", f"Input features saved to {csv_file}")

    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Create the main application window
app = ttk.Window(themename="litera")
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

# Correct variable type for workout type
workout_type_var = ttk.StringVar(value="HIIT")  # Default to "HIIT"
ttk.Label(app, text="Workout Type:", font=("Arial", 12)).grid(row=1, column=1, sticky="w", padx=20, pady=5)
ttk.Radiobutton(app, text="HIIT", variable=workout_type_var, value="HIIT", bootstyle="success").grid(row=2, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Yoga", variable=workout_type_var, value="Yoga", bootstyle="success").grid(row=3, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Cardio", variable=workout_type_var, value="Cardio", bootstyle="success").grid(row=4, column=1, sticky="w", padx=40, pady=5)
ttk.Radiobutton(app, text="Strength", variable=workout_type_var, value="Strength", bootstyle="success").grid(row=5, column=1, sticky="w", padx=40, pady=5)

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
    command=save_input_vector_to_dataframe, 
    bootstyle="primary"
).grid(row=22, column=0, columnspan=2, pady=30, sticky="w", padx=300)

# Run the application
app.mainloop()