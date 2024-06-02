import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score



class ClassificationComparer:
    def __init__(self):
        self.models = {}
        self.hyperparams = {}
        self.trained_models = {}

    def add_model(self, name, model, hyperparams):
        """
        Add a new model to the comparer.
        
        Parameters:
        - name: str, name of the model
        - model: scikit-learn estimator instance, the machine learning model
        - hyperparams: dict, hyperparameters for tuning the model
        """
        self.models[name] = model
        self.hyperparams[name] = hyperparams

    def tune_and_train(self, X_train, X_test, y_train, y_test):
        """
        Tune hyperparameters and train the models.
        
        Parameters:
        - X: array-like, feature matrix
        - y: array-like, target vector
        - test_size: float, proportion of the dataset to include in the test split
        - random_state: int, seed used by the random number generator
        """
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        for name in self.models:
            print(f"Training and tuning {name}...")
            model = self.models[name]
            hyperparams = self.hyperparams[name]
            
            grid_search = GridSearchCV(model, hyperparams, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_model.fit(X_train, y_train)
            
            self.trained_models[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'train_score': best_model.score(X_train, y_train),
                'test_score': best_model.score(X_test, y_test)
            }

    def predict(self, name, X):
        """
        Make predictions using the specified model.
        
        Parameters:
        - name: str, name of the model to use for predictions
        - X: array-like, feature matrix
        
        Returns:
        - array-like, predictions
        """
        if name not in self.trained_models:
            raise ValueError(f"Model {name} is not trained yet.")
        
        model = self.trained_models[name]['model']
        return model.predict(X)

    def compare_models(self):
        """
        Compare the performance of the trained models.
        
        Returns:
        - dict, performance metrics of the trained models
        """
        comparison = {}
        for name, details in self.trained_models.items():
            comparison[name] = {
                'best_params': details['best_params'],
                'train_score': details['train_score'],
                'test_score': details['test_score']
            }
        return comparison