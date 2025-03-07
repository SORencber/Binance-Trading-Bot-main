import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib   # model kaydetmek/yüklemek için
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,   r2_score

from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, RFE

#=========================================================
# 1) Veri Hazırlama
#=========================================================
def prepare_data(df):
    """
    Temel veri temizleme, missing value doldurma, object kolonları atma vs.
    """
    df = df.copy()
    # fillna
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    df = df.fillna(df.median(numeric_only=True))
    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop non-numeric
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_cols, errors='ignore')
    # Tekrar fillna
    df = df.fillna(df.median(numeric_only=True))
    return df

#=========================================================
# 2) Feature Selection Fonksiyonları
#=========================================================
def select_high_variance_features(X, threshold=0.01):
    var_thresh = VarianceThreshold(threshold=threshold)
    X_high_variance = var_thresh.fit_transform(X)
    selected_cols = X.columns[var_thresh.get_support()]
    return pd.DataFrame(X_high_variance, columns=selected_cols)

def select_using_mutual_information(X, y, top_n=10):
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MutualInfo': mi_scores
    }).sort_values(by='MutualInfo', ascending=False)
    # top_n feature
    top_features = mi_df.head(top_n)['Feature']
    return X[top_features]

def select_using_rfe(X, y, model, top_n=10):
    selector = RFE(model, n_features_to_select=top_n)
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_]
    return X[selected_cols]

#=========================================================
# 3) Model Eğitimi ve Değerlendirme
#=========================================================
def train_and_evaluate_models(X, y):
    """
    - 3 farklı model (RF, GBoost, Linear)
    - TimeSeriesSplit ile MAE hesaplar
    - En düşük hatalı modeli 'best_model' olarak döndürür
    """
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
        "GradientBoost": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "Linear": LinearRegression()
    }
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    lowest_error = float('inf')
    best_y_pred = None
    final_test_y = None

    for model_name, model in models.items():
        errors = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = np.mean(np.abs(y_test - preds))
            r2   = r2_score(y_test, preds)

            errors.append(mae)

            # en iyi ise kaydet
            if mae < lowest_error:
                best_model = (model_name, model)
                lowest_error = mae
                best_y_pred = preds
                final_test_y = y_test


        print(f"{model_name} => Mean MAE: {np.mean(errors):.4f}")

    # Ek metrik
    mse  = mean_squared_error(final_test_y, best_y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(final_test_y, preds)

    print(f"Best Model: {best_model[0]}")
    print(f"MAE={lowest_error:.4f}, RMSE={rmse:.4f},R2={r2:.4f}")
    return best_model, lowest_error

#=========================================================
# 4) Model Kaydetme / Yükleme
#=========================================================
def save_model_and_columns(model, columns, model_path="1m_my_model.pkl", cols_path="final_columns.npy"):
    """
    best_model[1] => actual model object
    columns => X_final.columns
    """
    joblib.dump(model, model_path)
    np.save(cols_path, columns)  # columns is Index or list

def load_model_and_columns(model_path="my_model.pkl", cols_path="final_columns.npy"):
    loaded_model = joblib.load(model_path)
    final_cols  = np.load(cols_path, allow_pickle=True)
    return loaded_model, final_cols

#=========================================================
# 5) Tek Satır / Yeni Veri Tahmini
#=========================================================
def predict_new_data(model, df_new, final_columns):
    """
    df_new: 1 satır ya da az veri, 
    final_columns: eğitimde seçtiğimiz feature listesi
    """
    # df_new => prepare_data
    df_new_prep = prepare_data(df_new)

    # EĞER: feature selection adımlarında "fit" yapılan kısımları 
    # pipeline'a çevirmeniz lazımdı. Burada basit yaklaşımla:
    # "df_new_prep[final_columns]" => 
    # so long as final_columns is consistent with 
    # the final data in training

    X_new = df_new_prep[final_columns].values  # shape=(1, n_features)
    y_pred = model.predict(X_new)
    return y_pred

#=========================================================
# 6) Tüm Pipeline
#=========================================================
def main_pipeline(df):
    """
    - df hazırlama
    - X,y ayırma
    - Feature select
    - Model train
    - Kaydet
    """
    df_prep = prepare_data(df)

    # Hedef = df['Close']
    # Basit senaryo => drop some columns
    y = df_prep['close']
    # Örnek: X = df_prep den "Close" ve "High","Low" gibi istenmeyen kolonları at
    drop_cols = ['close','high','low','open','volume','close time']
    X = df_prep.drop(columns=drop_cols, errors='ignore')

    # 1) high variance
    X_hv = select_high_variance_features(X)
    # 2) mutual info
    X_mi = select_using_mutual_information(X_hv, y, top_n=15)
    # 3) RFE
    # or skip?
    X_final = select_using_rfe(X_mi, y, model=LinearRegression(), top_n=10)

    best_model, lowest_error = train_and_evaluate_models(X_final, y)

    # Kaydet
    model_obj = best_model[1]
    save_model_and_columns(model_obj, X_final.columns, "my_model.pkl", "final_columns.npy")

    return model_obj, X_final.columns

#=========================================================
# Örnek Kullanım
#=========================================================
if __name__ == "__main__":
    # 1) CSV oku
    symbol="BTCUSDT"
  
    csv_path = f"{symbol}_data.csv"
    df_main = pd.read_csv(csv_path)

    # 2) Tüm pipeline => model eğit, kaydet
    model_obj, final_cols = main_pipeline(df_main)
    print("Model eğitimi tamam ve kaydedildi.")

    # 3) Daha sonra / Farklı bir kod bloğunda "yeni veri" geldiğini düşünelim
    #    Bu "yeni veri" => single row / few rows
    loaded_model, loaded_cols = load_model_and_columns("my_model.pkl", "final_columns.npy")

    # diyelim son 1 satırı aldık (veya cımbızla = df_new)
    df_new = df_main.iloc[[-1]].copy()  # son satır
    y_pred_value = predict_new_data(loaded_model, df_new, loaded_cols)
    print("Tek satır tahmini:", y_pred_value)
    # eğer y_pred_value > df_new['Close'].iloc[0] => up
    # else => down
