import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix,  precision_recall_curve, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def main():
    # PART 1: Import Data, Preprocess, Train Model, Evaluate Performance
    df = importRiceDataFrame()

    # Basic data check: .head() .shape .info() .duplicated().sum() .describe()
    checkData(df)
    
    # Split data into X (features) and y (labels)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Check how many instances each class has and range of each feature
    checkClassImbalance(y)

    # Apply Label Encoding for Class column (Osmancik -> [1,0], Cammeo -> [0,1])
    y = LabelEncoder().fit_transform(y)

    # Split data into training and testing sets 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train the model using Random Forest Classifier
    model = trainModel(X_train, y_train)
    # Compute F1 Score on full dataset
    full_f1 = f1_score(y_test, model.predict(X_test))

    # Evaluate performance
    generateF1ScoreReport(model, X_test, y_test)
    generateConfusionMatrix(model, X_test, y_test)
    generatePrecisionRecallCurve(model, X_test, y_test)

    # PART 2: Compute Brute Force Leave-One-Out Influence From Randomly Selected 10 Indexes
    computeLooInfluence(full_f1, X_train, y_train, X_test, y_test)

    # PART 3: Compute Group-Level Influence From 10 Groups Increasing In Size (10%,20%,...,100%)
    computeGroupInfluence(full_f1, X_train, y_train, X_test, y_test)

    # Part 4: Compute Shapley Values Using Monte Carlo 
    computeShapleyValues(full_f1, X_train, y_train, X_test, y_test)
    

def importRiceDataFrame():
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
    df = pd.concat([rice_cammeo_and_osmancik.data.features, rice_cammeo_and_osmancik.data.targets], axis=1)
    return df

def checkData(df):
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.duplicated().sum())
    print(df.describe())

def checkClassImbalance(y):
    print(y.value_counts())

def generateF1ScoreReport(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("////// F1 Score Report //////")
    print(classification_report(y_test, y_pred))

def generateConfusionMatrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def generatePrecisionRecallCurve(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_pred)
    
    sns.lineplot(x=recall, y=precision, marker='o', label='AP Score: '+ str(ap_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

def trainModel(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs = -1)
    model.fit(X_train, y_train)
    return model

def computeLooInfluence(full_f1, X_train, y_train, X_test, y_test):
    # Select 10 random and unique indexes from the training data
    selected_indices = np.random.choice(len(X_train), 10, replace=False)

    loo_influences = []

    # Compute LOO influence for each selected index
    for i in selected_indices:
        X_train_subset = np.delete(X_train, i, axis=0)
        y_train_subset = np.delete(y_train, i, axis=0)
    
        model_subset = trainModel(X_train_subset, y_train_subset)
        subset_f1 = f1_score(y_test, model_subset.predict(X_test))
    
        influence = subset_f1 - full_f1
        loo_influences.append((i, influence))
        print("Influence Score for Index: ", i, " is ", influence)

    loo_df = pd.DataFrame(loo_influences, columns=['Index', 'Influence'])

    sns.barplot(x='Index', y='Influence', data=loo_df)
    plt.xlabel('Index')
    plt.ylabel('Influence on F1-score')
    plt.axhline(0, color='black', linestyle='-')
    plt.show()

def computeGroupInfluence(full_f1, X_train, y_train, X_test, y_test):
    sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    group_influences = []

    # Compute Group influence for each selected fraction of data
    for i in sample_sizes:
        X_train_subset = pd.DataFrame(X_train).sample(frac=i).squeeze()
        y_train_subset = pd.DataFrame(y_train).sample(frac=i).squeeze()

        subset_size = len(X_train_subset)
    
        model_subset = trainModel(X_train_subset, y_train_subset)
        subset_f1 = f1_score(y_test, model_subset.predict(X_test))
    
        influence = subset_f1 - full_f1
        group_influences.append((subset_size, influence))
        print("Influence Score for Group Size: ", subset_size, " is ", influence)

    group_df = pd.DataFrame(group_influences, columns=['Sample Size', 'Influence'])

    sns.barplot(x='Sample Size', y='Influence', data=group_df)
    plt.xlabel('Group Size')
    plt.ylabel('Influence on F1-score')
    plt.axhline(0, color='black', linestyle='-')
    plt.show()

def V(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs = -1)
    model.fit(X_train, y_train)
    score = f1_score(y_test, model.predict(np.array(X_test)))
    return score

def computeShapleyValues(full_f1, X_train, y_train, X_test, y_test, num_permutations=10, performance_tolerance=0.001):
    n = len(X_train)
    shapley_values = np.zeros(n)
    t = 0

    while t < num_permutations:
        t += 1
        random_indices = np.random.permutation(n)
        marginal_contributions = np.zeros(n)
        subset_X = np.empty((0, len(X_train.columns)))
        subset_y = np.empty(0)
        prev_score = 0

        for j, i in enumerate(random_indices):
            subset_X = np.vstack([subset_X, X_train.iloc[i].values.reshape(1, -1)])
            subset_y = np.append(subset_y, y_train[i])

            if j > 0:
                new_score = V(subset_X, subset_y, X_test, y_test)
                marginal_contribution = new_score - prev_score
                
                if performance_tolerance > abs(full_f1 - new_score):
                    break
                
                marginal_contributions[i] = marginal_contribution
                prev_score = new_score

        shapley_values += marginal_contributions / num_permutations

    plt.hist(shapley_values, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Shapley Value')
    plt.ylabel('Frequency')
    plt.title('Estimated Shapley Values')
    plt.show()

main()