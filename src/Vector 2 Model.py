import pickle

# Define the file paths for your models
model_files = [
    'models/gender_stats.pkl',
    'models/kmeans_model.pkl',
    'models/numerical_scaler.pkl',
    'models/one_hot_encoder.pkl',
    'models/scaled_columns.pkl'
]

# Load the models
models = []
for file in model_files:
    with open(file, 'rb') as f:
        models.append(pickle.load(f))

# Now you can use the models list to access your loaded models
for i, model in enumerate(models):
    print(f"Model {i+1} loaded successfully")