import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def preprocess_and_check(df):
    # Forward and backward fill missing values
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    
    # Select only numeric columns for infinity check
    numeric_df = df_filled.select_dtypes(include=[np.number])
    
    # Check for infinity values
    if np.isinf(numeric_df.values).any():
        print("Infinity values found. Replacing with NaN.")
        df_filled.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN values again
        df_filled.fillna(method='ffill', inplace=True)
        df_filled.fillna(method='bfill', inplace=True)
    
    # Check for extremely large values
    max_value = numeric_df.values.max()
    if max_value > 1e6:  # Threshold for large values
        print(f"Extremely large values found (max: {max_value}). Consider scaling the data.")
    
    return df_filled

def evaluate_models_with_decision(df):
    """
    Evaluate models, predict next 15-minute price direction, and return the last Close value and final decision.
    """
    df_filled = preprocess_and_check(df)
    df_filled['close_Change'] = df_filled['close'].shift(-1) - df_filled['close']
    df_filled['Price_Direction'] = (df_filled['close_Change'] > 0).astype(int)
    df_filled = df_filled[:-1]  # Remove last row with NaN target
    
    features = df_filled.drop(columns=['close_Change', 'Price_Direction', 'close time'])
    target = df_filled['Price_Direction']
    
    # Standardize the features for models sensitive to scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tscv = TimeSeriesSplit(n_splits=10)
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=50),
        'Logistic Regression': LogisticRegression(random_state=50, max_iter=2000),
        'SVM': SVC(probability=True, random_state=50),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=10),
        'Decision Tree': DecisionTreeClassifier(random_state=50)
    }
    
    results = []
    for name, clf in classifiers.items():
        validation_accuracies = []
        for train_index, val_index in tscv.split(features_scaled):
            X_train, X_val = features_scaled[train_index], features_scaled[val_index]
            y_train, y_val = target.iloc[train_index], target.iloc[val_index]
            
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            validation_accuracies.append(val_accuracy)
        
        avg_accuracy = np.mean(validation_accuracies)
        latest_prediction = clf.predict(features_scaled[-1].reshape(1, -1))[0]
        prediction_direction = "Up" if latest_prediction == 1 else "Down"
        
        results.append((name, prediction_direction, avg_accuracy))
    
    # Decision mechanism: Aggregate results
    up_votes = sum(1 for _, pred, acc in results if pred == "Up" and acc > 0.5)
    down_votes = sum(1 for _, pred, acc in results if pred == "Down" and acc > 0.5)
    
    final_decision = 1 if up_votes > down_votes else 0
    last_close_value = df_filled['close'].iloc[-1]
    last_close_date = df_filled['close time'].iloc[-1]
    
    # Print results and decision
    for name, pred, acc in results:
        print(f"Model: {name}, Predicted Next : {pred}, Accuracy: {acc:.2f}")
        print(f"Date : {last_close_date}")

    print(f"\nLast Close Value: {last_close_value}")
    print(f"Final Decision Based on Models: {final_decision}")
    
    #return last_close_value, final_decision
if __name__ == '__main__':
    # Model dosyası yoksa ilk eğitimi 
    symbol="BTCUSDT"
  
    csv_path = f"{symbol}_data.csv"
    data = pd.read_csv(f"{symbol}_data.csv", parse_dates=["timestamp"])
    data.sort_values("timestamp", inplace=True)
    data = data.ffill().bfill()  # Eksik verileri doldur
    data = data[data['close'] > 0]  # Geçersiz değerleri temizle
    # Geçmiş bir tarih aralığı için tahmin yap
    # Eğitimde kullanılmayacak sütunları çıkar (örneğin 'date')
    if 'date' in data.columns:
        data.drop('date', axis=1, inplace=True)

    evaluate_models_with_decision(data)
       
    