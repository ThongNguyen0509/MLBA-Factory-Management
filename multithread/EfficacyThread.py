from connector.Connector import Connector
import traceback
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal, QThread, QRunnable, QThreadPool, QVariant
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class ModelWorker(QThread):
   train_done = pyqtSignal(object)
   visualize_done = pyqtSignal()
   error_occurred = pyqtSignal(str)
   progress_updated = pyqtSignal(int)
   def __init__(self, selected_columns, parent=None):
       super().__init__(parent)
       self.selected_columns = selected_columns
       self.connector = Connector()
       self.columns_mapping = {
            'Gender':'sub_sex',
            'Age': 'sub_age',
            'Health':'sub_health_h',
            'Commitment':'sub_commitment_h',
            'Perceptiveness':'sub_perceptiveness_h',
            'Dexterxity':'sub_dexterity_h',
            'Sociality':'sub_sociality_h',
            'Goodness':'sub_goodness_h',
            'Strength': 'sub_strength_h',
            'Open-minded': 'sub_openmindedness_h',
            'Worker Supervisor Age Diff':'sup_sub_age_diff'            
        }
       self.cluster = None
   def connectdb(self):
       self.connector.server = "localhost"
       self.connector.port = 3306
       self.connector.database = "factorymanagement"
       self.connector.username = "root"
       self.connector.password = "@Obama123"
       self.connector.connect()

   def evaluate_models(self, X, y):
        models = [
            ("Random Forest", RandomForestRegressor()),
            ("XGBoost", XGBRegressor()),
            ("LightGBM", LGBMRegressor())
        ]

        best_model = None
        best_score = float('-inf')

        for name, model in models:
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
            if r2 > best_score:
                best_model = pipeline
                best_score = r2
        return best_model
   
   def run(self):
    try:
        self.connectdb()
        if not self.selected_columns:
            self.error_occurred.emit("Please select at least one column to train the model.")
            return

        sql = '''SELECT sub_age, sub_health_h, sub_commitment_h, sub_perceptiveness_h, sub_dexterity_h,
                sub_sociality_h, sub_goodness_h, sub_strength_h, sub_openmindedness_h, sup_sub_age_diff, actual_efficacy_h FROM factory WHERE record_comptype = "Efficacy"
            '''
        df = self.connector.queryDataset(sql)
        X = df[self.selected_columns]
        y = df['actual_efficacy_h']
        numerical_columns = [col for col in self.selected_columns]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), numerical_columns)
            ], remainder='passthrough'
        )

        total_steps = 4  
        step = 1
        self.progress_updated.emit(int(step / total_steps * 100))
        best_model = self.evaluate_models(X, y)
        step += 1
        self.progress_updated.emit(int(step / total_steps * 100))
        step += 1
        self.progress_updated.emit(int(step / total_steps * 100))
        step += 1
        self.progress_updated.emit(int(step / total_steps * 100))

        self.train_done.emit(best_model)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        self.error_occurred.emit(error_message)
        traceback.print_exc()


class VisWorker(QThread):
    vis_done = pyqtSignal(object)

    def __init__(self, model, selected_columns, parent=None):
        super().__init__(parent)
        self.model = model
        self.selected_columns = selected_columns
        self.connector = Connector()

    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()

    def run(self):
        try:
            self.connectdb()
            sql = '''SELECT sub_age, sub_health_h, sub_commitment_h, sub_perceptiveness_h, sub_dexterity_h, 
                sub_sociality_h, sub_goodness_h, sub_strength_h, sub_openmindedness_h, sup_sub_age_diff, actual_efficacy_h FROM factory WHERE record_comptype = "Efficacy"
            '''
            df = self.connector.queryDataset(sql)
            df = df.dropna()
            X = df[self.selected_columns]
            y = df['actual_efficacy_h']
            pipeline = self.model

            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            self.vis_done.emit((y, y_pred))

        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            traceback.print_exc()
