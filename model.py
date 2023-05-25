# Gerekli kütüphanelerin eklenmesi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# csv dosyalarının okunması
data = pd.read_csv('data.csv')
label = pd.read_csv('labels.csv')

input = data.iloc[:, 1:]
target = label.iloc[:, 1]


# Kanser türlerini sayısal etiketleme
label_mapping = {
    'colon cancer': 0,
    'lung cancer': 1,
    'breast cancer': 2,
    'prostate cancer': 3
}
target = target.map(label_mapping)

X = input.values.astype('float64')
y = target.values.astype('uint8')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Özellik ölçekleme
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Sınıflandırıcısı
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Confusion Matrix oluşturma (Random Forest)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# True Positives, False Positives, True Negatives, False Negatives değerlerinin hesaplanması (Random Forest)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf[0, 0], cm_rf[0, 1], cm_rf[1, 0], cm_rf[1, 1]

# Sensitivity, Specificity, ve Accuracy değerlerinin hesaplanması (Random Forest)
sensitivity_rf = tp_rf / (tp_rf + fn_rf)
specificity_rf = tn_rf / (tn_rf + fp_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Random Forest Results:")
print("Sensitivity:", sensitivity_rf)
print("Specificity:", specificity_rf)
print("Accuracy:", accuracy_rf)

# XGBoost Sınıflandırıcısı
xgb_classifier = XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

# Confusion Matrix hesaplanması (XGBoost)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)


# True Positives, False Positives, True Negatives, False Negatives değerlerinin hesaplanması (XGBoost)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb[0, 0], cm_xgb[0, 1], cm_xgb[1, 0], cm_xgb[1, 1]

# Sensitivity, Specificity, ve Accuracy değerlerinin hesaplanması (XGBoost)
sensitivity_xgb = tp_xgb / (tp_xgb + fn_xgb)
specificity_xgb = tn_xgb / (tn_xgb + fp_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Results:")
print("Sensitivity:", sensitivity_xgb)
print("Specificity:", specificity_xgb)
print("Accuracy:", accuracy_xgb)