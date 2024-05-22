from connector.Connector import Connector
from ui.ML_ui import Ui_MLWindow
from utils.FileUtil import FileUtil
import traceback
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QLabel, QLineEdit, QFrame, QListWidgetItem, QFormLayout, QPushButton, QProgressDialog, QProgressBar
from datetime import datetime
from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal, QThread
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
import threading

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
        categorical_columns = [col for col in self.selected_columns if col == 'sub_sex']
        numerical_columns = [col for col in self.selected_columns if col != 'sub_sex']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(), categorical_columns),
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

class MLEx(Ui_MLWindow):
    def __init__(self):
        self.connector = Connector()
        self.model = None
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

    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.b_trainEfficiency.clicked.connect(self.train_model)
        self.b_saveEfficiency.clicked.connect(self.save_model)
        self.b_loadEfficiency.clicked.connect(self.load_model)
        self.b_visualizeEfficiency.clicked.connect(self.visualize_efficiency_prediction)
        self.setupPlot()

    def setupPlot(self):
        figsize = (8, 6)
        self.figure = plt.figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.MainWindow)
        self.vl_efficiency.addWidget(self.toolbar)
        self.vl_efficiency.addWidget(self.canvas)

    def show(self):
        self.MainWindow.show()

    def get_selected_columns(self):
        selected_columns = []
        for i in range(self.lw_MLData.count()):
            item = self.lw_MLData.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ui_text = item.text()
                db_column_name = self.columns_mapping.get(ui_text, ui_text)
                selected_columns.append(db_column_name)
        return selected_columns
    
    def populate_column_list(self):
        self.lw_MLData.clear() 
        for column_name, db_column_name in self.columns_mapping.items():
            item = QListWidgetItem(column_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.lw_MLData.addItem(item)    

    def train_model(self):
        selected_columns = self.get_selected_columns()
        if not selected_columns:
            QMessageBox.warning(self.MainWindow, "Warning", "Please select at least one column to train the model.")
            return
        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.fl_modelUsage.addRow(self.progress_label, self.progress_bar)
        self.update_progress_bar()
        self.worker = ModelWorker(selected_columns)
        self.worker.train_done.connect(self.create_model_usage_ui)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.start()

    def update_progress_bar(self, value=0):
        self.progress_bar.setValue(value)

    def create_model_usage_ui(self, trained_pipeline):
        self.model = trained_pipeline

        while self.fl_modelUsage.rowCount() > 0:
            self.fl_modelUsage.removeRow(0)

        independent_columns = [col.replace('encoder__', '').replace('scaler__', '') for col in trained_pipeline[:-1].get_feature_names_out()]
        reverse_mapping = {v: k for k, v in self.columns_mapping.items()}
        independent_columns = [reverse_mapping.get(col, col) for col in independent_columns]
        for column_name in independent_columns:
            line_edit = QLineEdit()
            self.fl_modelUsage.addRow(column_name, line_edit)
        efficacy_label = QLabel("Predicted Efficacy:")
        predicted_value_label = QLabel("") 
        predicted_value_label.setStyleSheet("font-weight: bold;") 
        self.fl_modelUsage.addRow(efficacy_label, predicted_value_label)
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(lambda: self.update_predicted_value(predicted_value_label))
        self.fl_modelUsage.addRow(predict_button)

    def save_model(self):
        if not self.model:
            QMessageBox.warning(self.MainWindow, "Warning", "No trained model available to save.")
            return
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("Pickle files (*.pkl)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                if FileUtil.saveModel(self.model, file_path):
                    QMessageBox.information(self.MainWindow, "Success", "Model saved successfully.")
                else:
                    QMessageBox.warning(self.MainWindow, "Error", "Failed to save the model.")

    def on_error_occurred(self, error_message):
        QMessageBox.warning(self.MainWindow, "Error", error_message)

    def load_model(self):
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilter("Pickle files (*.pkl)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                loaded_pipeline = FileUtil.loadModel(file_path)
                if loaded_pipeline:
                    self.model = loaded_pipeline
                    QMessageBox.information(self.MainWindow, "Success", "Model loaded successfully.")
                    self.create_model_usage_ui(loaded_pipeline)
                    self.populate_column_list()
                else:
                    QMessageBox.warning(self.MainWindow, "Error", "Failed to load the model.")
                selected_columns = [col.replace('encoder__', '').replace('scaler__', '') for col in loaded_pipeline[:-1].get_feature_names_out()]
                for i in range(self.lw_MLData.count()):
                    item = self.lw_MLData.item(i)
                    column_name = item.text()
                    if column_name in self.columns_mapping.keys():
                        db_column_name = self.columns_mapping[column_name]
                        if db_column_name in selected_columns:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)


    def update_predicted_value(self, predicted_value_label):
        input_values = []
        for row in range(self.fl_modelUsage.rowCount() - 2):
            item = self.fl_modelUsage.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if item:
                input_value = item.widget().text()
                input_values.append(input_value)
        try:
            for i in range(len(input_values)):
                try:
                    input_values[i] = float(input_values[i])
                except ValueError:
                    pass  
            original_columns = [col.replace('encoder__', '').replace('scaler__', '') for col in self.model.named_steps['preprocessor'].get_feature_names_out()]
            input_data = {}
            for col_name, value in zip(original_columns, input_values):
                input_data[col_name] = [value]
            input_df = pd.DataFrame(input_data)
            transformed_input = self.model.named_steps['preprocessor'].transform(input_df)
            predicted_value = self.model.named_steps['regressor'].predict(transformed_input)
            predicted_value_label.setText(str(round(predicted_value[0],2)))
        except Exception as e:
            traceback.print_exc()

    def visualize_efficiency_prediction(self):
        if self.model is None:
            QMessageBox.warning(self.MainWindow, "Warning", "Please train the model first.")
            return
        selected_columns = self.get_selected_columns()
        self.vis_worker = VisWorker(self.model, selected_columns)
        self.vis_worker.vis_done.connect(self.update_vis_plot)
        self.vis_worker.start()


    def update_vis_plot(self, data):
        try:
            y, y_pred = data
            y = pd.to_numeric(y, errors='coerce')
            y_pred = pd.to_numeric(y_pred, errors='coerce')
            self.figure.clear()
            self.figure.set_size_inches(8, 3.8)
            ax = self.figure.add_subplot(111)
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.grid()
            x_interval = len(y) // 5
            y_min = min(min(y), min(y_pred))
            y_max = max(max(y), max(y_pred))
            y_interval = (y_max - y_min) / 10
            x_ticks = range(0, len(y), x_interval)
            y_ticks = [y_min + i * y_interval for i in range(5)]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.plot(x_ticks, [y[i] for i in x_ticks], marker='o', linestyle='-', label='Actual Efficacy')
            ax.plot(x_ticks, [y_pred[i] for i in x_ticks], marker='', linestyle='--', label='Predicted Efficacy')
            ax.set_title('Efficacy Prediction')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Efficacy')
            ax.legend()
            self.canvas.draw()
        except AttributeError:
            QMessageBox.warning(self.MainWindow, "Warning", "Please select at least one column to train the model.")
        except Exception as e:
            print(e)
            print(y.dtypes,y_pred.dtypes)