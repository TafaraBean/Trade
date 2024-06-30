import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def calculate_vif(dataframe, feature_columns):
    X = dataframe[feature_columns]
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    return vif

def logistic_regression_accuracy(dataframe, feature_columns, target_column, test_size=0.2, random_state=42):
    # Extract features and target from the DataFrame
    X = dataframe[feature_columns]
    y = dataframe[target_column]

    # Remove features with zero variance
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    # Calculate VIF and remove highly correlated features
    vif = calculate_vif(dataframe, X.columns)
    high_vif_features = vif[vif['VIF Factor'] > 5]['Feature'].tolist()
    print(vif)
    #X = X.drop(columns=high_vif_features)

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Add a constant term for the intercept
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    # Initialize and train the logistic regression model
    model = sm.Logit(y_train, X_train)
    result = model.fit()
    
    # Predict the target for the test set
    y_pred = result.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    
    return accuracy, result.summary()

# Example usage with your data
data = pd.read_csv('beta/output.csv')

data['next_close'] = data['close'].shift(-1)
data['success2'] = (data['close'] < data['next_close']).astype(int)

data = data.dropna()

# Define the feature columns and the target column
feature_columns = ['prev_hour_lsma_slope','prev_hour_lsma']
target_column = 'success2'

# Calculate accuracy and get the model summary
accuracy, summary = logistic_regression_accuracy(data, feature_columns, target_column)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(summary)
