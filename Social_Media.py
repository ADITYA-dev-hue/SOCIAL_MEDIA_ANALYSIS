# ============================================================
# GEN Z SOCIAL MEDIA USAGE - ADDICTION LEVEL PREDICTION
# 1 MILLION ROW DATASET
# ============================================================


# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 2. Load Dataset

df = pd.read_csv(r"C:\Users\aditya prakash\Desktop\SOCIAL MEDIA TREND\genz_social_media_usage_1M.csv")

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# 3. Data Cleaning

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Remove rows where target is missing
df = df.dropna(subset=["addiction_level"])

print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

print("Dataset Shape After Cleaning:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# 4. Select Features and Target

features = [
    "age",
    "gender",
    "country",
    "daily_usage_hours",
    "primary_platform",
    "num_platforms_used",
    "purpose",
    "avg_session_minutes",
    "night_usage",
    "mental_health_score",
    "screen_time_before_sleep"
]

target = "addiction_level"

data = df[features + [target]].copy()

print("\n" + "=" * 80)
print("FEATURES AND TARGET")
print("=" * 80)

print("Input Features:")
for feature in features:
    print("-", feature)

print("\nTarget Variable:")
print("-", target)


# 5. Check Target Distribution

print("\n" + "=" * 80)
print("ADDICTION LEVEL DISTRIBUTION")
print("=" * 80)

print(data[target].value_counts())

print("\nPercentage Distribution:")

print(
    (data[target].value_counts(normalize=True) * 100)
    .round(2)
)


# 6. Handle Missing Values

categorical_columns = [
    "gender",
    "country",
    "primary_platform",
    "purpose",
    "night_usage"
]

numerical_columns = [
    "age",
    "daily_usage_hours",
    "num_platforms_used",
    "avg_session_minutes",
    "mental_health_score",
    "screen_time_before_sleep"
]

# Fill categorical missing values
for col in categorical_columns:
    data[col] = data[col].fillna("Unknown")

# Fill numerical missing values
for col in numerical_columns:
    data[col] = data[col].fillna(
        data[col].median()
    )

print("\n" + "=" * 80)
print("MISSING VALUES AFTER CLEANING")
print("=" * 80)

print(data.isnull().sum())


# 7. Encode Target Variable

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    data[target]
)

print("\n" + "=" * 80)
print("TARGET ENCODING")
print("=" * 80)

print("Target Classes:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)


# 8. Encode Categorical Features

X = data.drop(
    columns=[target]
)

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False
)

print("\n" + "=" * 80)
print("FEATURE ENCODING")
print("=" * 80)

print("Original Feature Count:", len(features))

print(
    "Encoded Feature Count:",
    X.shape[1]
)

print(
    "Encoded Dataset Shape:",
    X.shape
)


# 9. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

print(
    "Total Samples:",
    len(X)
)

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# 10. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# 11. Logistic Regression

print("\n" + "=" * 80)
print("MODEL 1 - LOGISTIC REGRESSION")
print("=" * 80)

logistic_model = LogisticRegression(max_iter=1000,class_weight="balanced")

logistic_model.fit(X_train_scaled,y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(y_test,logistic_pred)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 12. Decision Tree

print("\n" + "=" * 80)
print("MODEL 2 - DECISION TREE")
print("=" * 80)

decision_tree_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

decision_tree_precision = precision_score(
    y_test,
    decision_tree_pred,
    average="macro",
    zero_division=0
)

decision_tree_recall = recall_score(
    y_test,
    decision_tree_pred,
    average="macro",
    zero_division=0
)

decision_tree_f1 = f1_score(
    y_test,
    decision_tree_pred,
    average="macro",
    zero_division=0
)

from sklearn.inspection import DecisionBoundaryDisplay
X_plot = X_train[['avg_session_minutes', 'mental_health_score']]
y_plot = y_train

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_plot, y_plot)

DecisionBoundaryDisplay.from_estimator(
    dt_model,
    X_plot,
    response_method="predict",
    xlabel="Average Session Minutes",
    ylabel="Mental Health Score",
    grid_resolution=100,
    alpha=0.4
)

plt.scatter(
    X_plot.iloc[:, 0],
    X_plot.iloc[:, 1],
    c=pd.Categorical(y_plot).codes,
    edgecolors='black',
    s=10
)

plt.title("Decision Tree Decision Boundaries")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='High',
           markerfacecolor='yellow', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Low',
           markerfacecolor='purple', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Medium',
           markerfacecolor='teal', markersize=6)
]

plt.legend(handles=legend_elements, title='Addiction Level')
plt.show()

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree_model,
    feature_names=X_train.columns,
    class_names=[str(c) for c in decision_tree_model.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Structure - Addiction Level Prediction")
plt.tight_layout()
plt.show()

print("Accuracy :", round(decision_tree_accuracy, 4))
print("Precision:", round(decision_tree_precision, 4))
print("Recall   :", round(decision_tree_recall, 4))
print("F1 Score :", round(decision_tree_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        decision_tree_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 13. Random Forest

print("\n" + "=" * 80)
print("MODEL 3 - RANDOM FOREST")
print("=" * 80)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_pred = random_forest_model.predict(
    X_test
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

random_forest_precision = precision_score(
    y_test,
    random_forest_pred,
    average="macro",
    zero_division=0
)

random_forest_recall = recall_score(
    y_test,
    random_forest_pred,
    average="macro",
    zero_division=0
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_pred,
    average="macro",
    zero_division=0
)

print("Accuracy :", round(random_forest_accuracy, 4))
print("Precision:", round(random_forest_precision, 4))
print("Recall   :", round(random_forest_recall, 4))
print("F1 Score :", round(random_forest_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        random_forest_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 14. K-Nearest Neighbors

print("\n" + "=" * 80)
print("MODEL 4 - K-NEAREST NEIGHBORS")
print("=" * 80)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn_model.predict(
    X_test_scaled
)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

knn_precision = precision_score(
    y_test,
    knn_pred,
    average="macro",
    zero_division=0
)

knn_recall = recall_score(
    y_test,
    knn_pred,
    average="macro",
    zero_division=0
)

knn_f1 = f1_score(
    y_test,
    knn_pred,
    average="macro",
    zero_division=0
)

print("Accuracy :", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall   :", round(knn_recall, 4))
print("F1 Score :", round(knn_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        knn_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 15. Compare All Four Models

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],

    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        random_forest_accuracy,
        knn_accuracy
    ],

    "Precision": [
        logistic_precision,
        decision_tree_precision,
        random_forest_precision,
        knn_precision
    ],

    "Recall": [
        logistic_recall,
        decision_tree_recall,
        random_forest_recall,
        knn_recall
    ],

    "F1 Score": [
        logistic_f1,
        decision_tree_f1,
        random_forest_f1,
        knn_f1
    ]
})


# 16. Sort Models by F1 Score

results = results.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# 17. Print Final Model Comparison

print("\n\n")

print("=" * 100)
print("                         FINAL MODEL COMPARISON")
print("=" * 100)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

print("=" * 100)


# 18. Find Best Model

best_model = results.iloc[0]

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "Model     :",
    best_model["Model"]
)

print(
    "Accuracy  :",
    round(best_model["Accuracy"], 4)
)

print(
    "Precision :",
    round(best_model["Precision"], 4)
)

print(
    "Recall    :",
    round(best_model["Recall"], 4)
)

print(
    "F1 Score  :",
    round(best_model["F1 Score"], 4)
)


# 19. Accuracy Comparison Graph

plt.figure(figsize=(10, 6))

bars = plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title(
    "Addiction Level Classification - Accuracy Comparison"
)

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy")

plt.xticks(
    rotation=20
)

plt.ylim(
    0,
    1
)

for bar, value in zip(
    bars,
    results["Accuracy"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.show()


# 20. F1 Score Comparison Graph

plt.figure(figsize=(10, 6))

bars = plt.bar(
    results["Model"],
    results["F1 Score"]
)

plt.title(
    "Addiction Level Classification - F1 Score Comparison"
)

plt.xlabel("Machine Learning Model")

plt.ylabel("Macro F1 Score")

plt.xticks(
    rotation=20
)

plt.ylim(
    0,
    1
)

for bar, value in zip(
    bars,
    results["F1 Score"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.show()


# 21. Confusion Matrices

predictions = {

    "Logistic Regression":
        logistic_pred,

    "Decision Tree":
        decision_tree_pred,

    "Random Forest":
        random_forest_pred,

    "KNN":
        knn_pred
}


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 11)
)


for ax, (model_name, prediction) in zip(
    axes.ravel(),
    predictions.items()
):

    cm = confusion_matrix(
        y_test,
        prediction
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        model_name
    )


plt.tight_layout()

plt.show()


# 22. Random Forest Feature Importance

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        random_forest_model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 80)
print("TOP 20 RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# 23. Feature Importance Graph

top_features = feature_importance.head(15)


plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top 15 Features for Addiction Level Prediction"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# 24. Addiction Level Distribution Graph

plt.figure(figsize=(9, 6))

addiction_counts = data[
    target
].value_counts()

bars = plt.bar(
    addiction_counts.index.astype(str),
    addiction_counts.values
)

plt.title(
    "Distribution of Addiction Levels"
)

plt.xlabel(
    "Addiction Level"
)

plt.ylabel(
    "Number of Users"
)

for bar, value in zip(
    bars,
    addiction_counts.values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.show()


# 25. Final Summary

print("\n")

print("=" * 80)
print("                    FINAL ANALYSIS SUMMARY")
print("=" * 80)

print(
    "Total Records Used:",
    len(data)
)

print(
    "Training Records:",
    len(X_train)
)

print(
    "Testing Records:",
    len(X_test)
)

print(
    "Number of Features After Encoding:",
    X.shape[1]
)

print(
    "Number of Classification Models:",
    4
)

print(
    "Best Model:",
    best_model["Model"]
)

print(
    "Best Model Accuracy:",
    round(
        best_model["Accuracy"],
        4
    )
)

print(
    "Best Model Macro F1 Score:",
    round(
        best_model["F1 Score"],
        4
    )

)

print("=" * 80)

print(
    "ADDICTION LEVEL CLASSIFICATION COMPLETED"
)

print("=" * 80)