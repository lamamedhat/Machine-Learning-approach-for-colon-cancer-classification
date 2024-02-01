import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score , f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split , GridSearchCV, KFold, cross_val_predict
from lazypredict.Supervised import LazyClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import sys
import os

def read_csv_to_dataframe(csv_path):
    """
    Read data from a CSV file into a pandas DataFrame.
     
    Parameters:
    - csv_path (str): Path to a CSV file.
    
    Returns:
    - Pd.DataFrame: The DataFrame containig the data.
    """
    try:
        df=pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading csv file: {e}")
        sys.exit(1)

def Correlation_between_features(dataframe):
    """
    Generate Correlation Matrix for the given DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe for which to generate correlation Matrix.

    Returns:
    - Correlation Matrix between Features. 
    
    """
    try:
        correlation = dataframe.corr()
        return correlation
    except Exception as e:
        print(f"Error Generating Correlation Matrix between features {e}")
        sys.exit(1)

        
def Filter_Correlation_Using_Threshold(dataset, threshold):
    
    """
    Filter the correlated columns from a dataset based on a specified threshold.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to filter.
    - threshold (float): The correlation threshold to use for filtering.

    Returns:
    - A set containing the names of the correlated columns.

    """
    try:
        col_corr = set()  
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: 
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    except Exception as e:
        print(f"Error Filtering correlated columns {e}")
        sys.exit(1)
        
    
    
def best_classifier(x,y,split_size):
    """
    Determine the best classifier using the Lazypredict Algorithm.
    
    Parameters:
    - x (pd.DataFrame) : The features of your dataframe.
    - y (pd.DataFrame) : The Label of your dataframe.
    
    Returns:
    - The best classifier based on Accuracy.
    
    """
    try:
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=split_size, random_state=42)
        clf = LazyClassifier(verbose=0, ignore_warnings=True,custom_metric=None)
        models,predictions=clf.fit(x_train,x_test,y_train,y_test)
    
        best_classifier = models.loc[models['Accuracy'].idxmax()]
        return best_classifier
    
    except Exception as e:
        print(f"Error Generating best classifier {e}")
        sys.exit(1)


def hyperparameter_KFold_CV(x,y):

    """
    Perform hyperparameter tuning using K-Fold cross-validation.

    Parameters:
    - x (pd.DataFrame) : The features of your dataframe.
    - y (pd.DataFrame) : The Label of your dataframe.

    Returns:
    - The best model according to the hyperparameters.
    
    """
    global KFoldCV
    global best_estimator
    global model
    global grid_search
    global param_grid
    try:
        param_grid={
                "criterion":['gini','entropy'],
                "max_depth":range(1,10),
                "max_features" : ['auto', 'sqrt']}

        model = DecisionTreeClassifier(random_state=1)
        KFoldCV = KFold(n_splits=5, shuffle=True, random_state=42) 
        grid_search = GridSearchCV(model, param_grid, cv=KFoldCV)
        grid_search.fit(x, y)
        best_estimator = grid_search.best_estimator_
        joblib.dump(best_estimator, 'best_estimator.pkl')
        return best_estimator

    except Exception as e:
        print(f" Error Generating The best estimator{e}")    
        sys.exit(1)
        
        
def hyperparameter_nested_KFold(output_dir, x, y):
    """
    Write the accuracy, F1 score, and Roc curve of hyperparameter tuning 
    using nested K-Fold cross-validation to the specified output directory.

    Parameters:
    - output_dir (str): Path to the output directory.
    - x: The feature matrix.
    - y: The target variable.
    """
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

        for train_outer, test_outer in cv_outer.split(x, y):
            x_train_outer, x_test_outer = x[train_outer], x[test_outer]
            y_train_outer, y_test_outer = y[train_outer], y[test_outer]

            cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=cv_inner)
            grid_search.fit(x_train_outer, y_train_outer)

            y_pred_outer = grid_search.predict(x_test_outer)

            # Calculate accuracy
            accuracy = accuracy_score(y_test_outer, y_pred_outer)
            print(f"Accuracy of nested_KFold_CV:{accuracy}")

            # Calculate F1 score
            f1 = f1_score(y_test_outer, y_pred_outer)
            print(f"f1 score of nested_KFold_CV:{f1}")

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test_outer, y_pred_outer)
            roc_auc = auc(fpr, tpr)
            print(f"Roc curve accuracy of nested_KFold_CV:{roc_auc}")

            # Plot the ROC curve
            plt.plot(fpr, tpr, label='ROC curve (average area = {:.2f})'.format(roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve of Nested KFold')
            plt.legend(loc="lower right")
            
            output = os.path.join(output_dir, "ROC_curve_of_Nested_KFold.jpg")
            plt.savefig(output)
            plt.clf()

    except Exception as e:
        print(f"Error writing Roc curve of hyperparameter nested KFold : {e}")
        sys.exit(1)
  

def write_correlation_to_directory(correlation,output_dir):

    """
    Write Correlation Matrix to the specified output directory.

    Parameters:
    - Correlation (pd.DataFrame): The Correlation Matrix between data Features.
    - output_dir (str): Path to the output directory.

    """
    try:
        output=os.path.join(output_dir,"HeatMap.jpg")
        plt.figure(figsize=(10,8), dpi =80)
        sns.heatmap(correlation,annot=True,fmt=".2f", linewidth=.5)
        plt.savefig(output)
        plt.clf()
    except Exception as e:
        print(f"Error Writing HeatMap of correlated features{e}")
        sys.exit(1)  

def write_hyperparameter_KFold_CV_to_directory(output_dir):
    """
    Write the Accuracy, F1_score and Roc curve of hyperparameter K-Fold cross-validation to the specified output directory.

    Parameters:
    - output_dir (str): Path to the output directory. 
    
    """
    try:
 
        # calculate accuracy
        ypred = cross_val_predict(best_estimator, x, y, cv=KFoldCV)
        accuracy = accuracy_score(y, ypred)
        print("Accuracy of KFold_CV:",accuracy)

        # calculate f1 score
        f1 = f1_score(y, ypred)
        print("F1 score of KFold_CV:", f1)
        
        # Roc curve
        fpr, tpr, thresholds = roc_curve(y, ypred)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc_curve KFold_CV')
        plt.legend(loc="lower right")
        
        output=os.path.join(output_dir,"Roc_curve of KFold_CV.jpg")
        plt.savefig(output)
        plt.clf()

    except Exception as e:
        print(f"Error Writing Roc_curve of hyperparameter KFold {e}")
        sys.exit(1)

         
 
if __name__=="__main__":
    if len(sys.argv)!=3:
        print(" Usage: python Script.py <csv_file> <output_directory> ")     
        sys.exit(1)   


csv_path=sys.argv[1]
output_dir=sys.argv[2]

# Read data into a DataFrame
data_df=read_csv_to_dataframe(csv_path)  

# Generate Correlation Matrix
correlation = Correlation_between_features(data_df)
    

# Filter the correlated columns based on a specified threshold
filtered_features = Filter_Correlation_Using_Threshold(data_df, 0.7)
filtered_data=data_df.drop(filtered_features, axis=1)
    
# Split the filtered data 
x=filtered_data.drop('class',axis=1)
y=filtered_data['class']

# Generate the best classifier using the Lazypredict Algorithm
best_classifier(x,y,0.3)

# Generate the best model according to the hyperparameters  
hyperparameter_KFold_CV(x,y)

numpy_x=np.asarray(x)
numpy_y=np.asarray(y)
# Generate the accuracy, F1_score and Roc curve of hyperparameter nested KFold Cross-Validation
hyperparameter_nested_KFold(output_dir,numpy_x,numpy_y)

# Generate Correlation Matrix
write_correlation_to_directory(correlation,output_dir)

# Generate the accuracy, F1_score and Roc curve of hyperparameter KFold Cross-Validation
write_hyperparameter_KFold_CV_to_directory(output_dir)


