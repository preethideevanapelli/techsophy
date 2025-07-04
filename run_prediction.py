import pandas as pd
from feature_engineering import create_features
from model import train_model, load_model, predict
from intervention import apply_interventions
from scheduling import optimize_schedule

def main():
    # Load your data here
    df = pd.read_csv('appointments.csv')
    
    # Feature engineering
    df, encoders = create_features(df)
    
    # Prepare features and target
    X = df.drop(columns=['no_show', 'appointment_day'])
    y = df['no_show']
    
    # Train model (or load pre-trained)
    model = train_model(X, y)
    
    # Predict on same data or test set
    proba, pred = predict(model, X)
    df['no_show_prob'] = proba
    df['predicted_no_show'] = pred
    
    # Apply interventions
    df = apply_interventions(df)
    
    # Optional: Optimize scheduling
    df = optimize_schedule(df)
    
    # Save results
    df.to_csv('predictions.csv', index=False)
    print("Pipeline complete. Predictions saved.")

if __name__ == "__main__":
    main()
