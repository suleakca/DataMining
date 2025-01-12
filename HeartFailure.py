import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart.csv")

print(df.info())

print(df.columns)
# Checking for duplicate rows 
duplicates = df.duplicated()
print("Number of duplicate rows: ", duplicates.sum())
print("Number of null values")
print(df. isnull(). sum())

# Checking for categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)

# Checking if there are any categorical variables, use one-hot encoding
if len(categorical_columns) > 0:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


print(df.info())

# correlation matrix
corr_matrix = df.corr()
# heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Heart Failure')
plt.show()

# barplot of the target variable
sns.countplot(x='HeartDisease', data=df)
plt.title('Distribution of Heart Disease')
plt.show()


sns.boxplot(x='HeartDisease', y='Age', data=df)
plt.title('Age by Heart Disease')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='HeartDisease', y='Age', data=df, inner="quartile")

plt.xlabel('Heart Disease')
plt.ylabel('Age')
plt.title('Age Distribution by Heart Disease')
plt.show()

sns.boxplot(x='HeartDisease', y='RestingBP', data=df)
plt.title('Resting Blood Pressure by Heart Disease')
plt.show()

sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
plt.title('Cholesterol by Heart Disease')
plt.show()

sns.boxplot(x='HeartDisease', y='FastingBS', data=df)
plt.title('Fasting Blood Sugar by Heart Disease')
plt.show()

sns.boxplot(x='HeartDisease', y='MaxHR', data=df)
plt.title('Maximum Heart Rate by Heart Disease')
plt.show()

sns.boxplot(x='HeartDisease', y='Oldpeak', data=df)
plt.title('Oldpeak by Heart Disease')
plt.show()



X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)


print("Gradient Boosting Predict:", y_pred_gbc)
print("Random Forest Predict:", y_pred_rf)
print("Decision Tree Predict:", y_pred_dtree)

# Gradient Boosting Evaluation
print("Gradient Boosting Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_gbc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))
print("Classification Report:\n", classification_report(y_test, y_pred_gbc))

# Random Forest Evaluation
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Decision Tree Evaluation
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_dtree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dtree))
print("Classification Report:\n", classification_report(y_test, y_pred_dtree))

# Feature Importance for Gradient Boosting
feature_importance_gbc = gbc.feature_importances_
features = X.columns
importance_df_gbc = pd.DataFrame({'Feature': features, 'Importance': feature_importance_gbc})
importance_df_gbc = importance_df_gbc.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_gbc)
plt.title('Gradient Boosting Feature Importance')
plt.show()

# Feature Importance for Random Forest
feature_importance_rf = rf.feature_importances_
importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': feature_importance_rf})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_rf)
plt.title('Random Forest Feature Importance')
plt.show()

# Feature Importance for Decision Tree
feature_importance_dtree = dtree.feature_importances_
importance_df_dtree = pd.DataFrame({'Feature': features, 'Importance': feature_importance_dtree})
importance_df_dtree = importance_df_dtree.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_dtree)
plt.title('Decision Tree Feature Importance')
plt.show()

# Compare the accuracies
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)

print(f"Gradient Boosting Accuracy: {accuracy_gbc}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Decision Tree Accuracy: {accuracy_dtree}")

# best model based on accuracy
best_model = "Gradient Boosting" if accuracy_gbc > accuracy_rf and accuracy_gbc > accuracy_dtree else \
             "Random Forest" if accuracy_rf > accuracy_dtree else \
             "Decision Tree"
print(f"The best model for this dataset is: {best_model}")
