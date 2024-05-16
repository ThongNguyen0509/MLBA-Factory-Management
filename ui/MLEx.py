from connector.Connector import Connector
from ui.ML_ui import Ui_MLWindow
from utils.FileUtil import FileUtil
import traceback
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
           if not self.selected_columns:
               self.error_occurred.emit("Please select at least one column to train the model.")
               return

           sql = "SELECT * FROM factory WHERE record_comptype = 'Efficacy'"
           df = self.connector.queryDataset(sql)
           X = df[self.selected_columns]
           y = df['actual_efficacy_h']
           categorical_columns = [col for col in self.selected_columns if col == 'sub_sex']
           numerical_columns = [col for col in self.selected_columns if col != 'sub_sex']
           preprocessor = ColumnTransformer(
               transformers=[
                   ('encoder', OneHotEncoder(), categorical_columns),
                   ('scaler', StandardScaler(), numerical_columns)
               ], remainder='passthrough'
           )
           model = RandomForestRegressor()
           pipeline = Pipeline([
               ('preprocessor', preprocessor),
               ('regressor', model)
           ])
           total_steps = 3  
           step = 1
           self.progress_updated.emit(int(step / total_steps * 100))
           pipeline.fit(X, y)
           step += 1
           self.progress_updated.emit(int(step / total_steps * 100))
           step += 1
           self.progress_updated.emit(int(step / total_steps * 100))

           self.train_done.emit(pipeline)
       except Exception as e:
           error_message = f"An error occurred: {e}"
           self.error_occurred.emit(error_message)
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
        self.b_visualizeEfficiency.clicked.connect(self.VisualRandomForest)
        self.populate_column_list()
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
        for ui_text in self.columns_mapping.keys():
            item = QListWidgetItem(ui_text)
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

        independent_columns = trained_pipeline[:-1].get_feature_names_out()

        for column_name in independent_columns:
            line_edit = QLineEdit()
            self.fl_modelUsage.addRow(column_name.replace('scaler__',''), line_edit)
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
                else:
                    QMessageBox.warning(self.MainWindow, "Error", "Failed to load the model.")

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

    def VisualRandomForest(self):
        self.connectdb()
        selected_columns = self.get_selected_columns()
        if not selected_columns:
            raise ValueError("No columns selected for processing.")
        sql = "SELECT * FROM factory WHERE record_comptype = 'Efficacy'"
        df = self.connector.queryDataset(sql)
        X = df[selected_columns]
        y = df['actual_efficacy_h']
        pipeline = self.process_data_for_model()
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
        ax.set_title('Random Forest Regression')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        self.canvas.draw()