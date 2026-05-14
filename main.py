import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def carregamento_de_dados():
    caminho = "data/creditcard.csv"
    df = pd.read_csv(caminho)

    return df

def preprocessamento(df):
    df["amount_log"] = np.log1p(df["Amount"])

    scaler = StandardScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df.drop("Amount", axis=1, inplace=True)

    x = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.3, random_state=42
        )
    
    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, y_train_res, X_test, y_test

def treinamento(X_train_res, y_train_res, X_test, y_test):
    xgb = XGBClassifier( 
        scale_pos_weight=10,
        eval_metric="logloss"
    )

    xgb.fit(X_train_res, y_train_res)
    
    y_pred_xgb = xgb.predict(X_test)

    print(classification_report(y_test, y_pred_xgb))

    return xgb

def avaliacao(xgb, X_train_res):
    importancias = xgb.feature_importances_

    plt.barh(X_train_res.columns, importancias)
    plt.title("Importância das Variáveis")
    plt.show()

if __name__ == "__main__":
    df = carregamento_de_dados()
    X_train_res, y_train_res, X_test, y_test = preprocessamento(df)
    xgb = treinamento(X_train_res, y_train_res, X_test, y_test)
    avaliacao(xgb, X_train_res)
    