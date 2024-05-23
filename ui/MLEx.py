from multithread.EfficacyThread import VisWorker, ModelWorker
from connector.Connector import Connector
from ui.ML_ui import Ui_MLWindow
from utils.FileUtil import FileUtil
import traceback
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QLabel, QLineEdit, QFrame, QListWidgetItem, QFormLayout, QPushButton, QProgressBar, QMainWindow
from ui.ChangeInformationEx import ChangeInformation
from PyQt6.QtCore import Qt
import pandas as pd
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

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
            'Worker Supervisor Age Diff':'sup_sub_age_diff' , 
            'Efficacy':'efficacy'          
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
        self.b_loadEfficiency.clicked.connect(self.load_model_predict)
        self.b_visualizeEfficiency.clicked.connect(self.visualize_efficiency_prediction)
        self.b_SaveCluster.clicked.connect(self.save_model)
        self.b_TrainCluster.clicked.connect(self.perform_kmeans_clustering)
        self.b_LoadCluster.clicked.connect(self.load_cluster_model)
        self.b_changeProfile.clicked.connect(self.openChangeInfo)
        self.b_logout.clicked.connect(self.logout)
        self.setupPlotCluster()
        self.setupPlotPrediction()

    def setupPlotPrediction(self):
        figsize = (8, 6)
        self.figurePrediction = plt.figure(figsize=figsize)
        self.canvasPrediction = FigureCanvas(self.figurePrediction)
        self.toolbarPrediction = NavigationToolbar(self.canvasPrediction, self.MainWindow)
        self.vl_efficiency.addWidget(self.toolbarPrediction)
        self.vl_efficiency.addWidget(self.canvasPrediction)

    def setupPlotCluster(self):
        figsize = (8, 6)
        self.figureCluster = plt.figure(figsize=figsize)
        self.canvasCluster = FigureCanvas(self.figureCluster)
        self.toolbarCluster = NavigationToolbar(self.canvasCluster, self.MainWindow)
        self.vl_VisualizeCluster.addWidget(self.toolbarCluster)
        self.vl_VisualizeCluster.addWidget(self.canvasCluster)

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
        button = self.MainWindow.sender()
        if button == self.b_saveEfficiency:
            if not self.model:
                QMessageBox.warning(self.MainWindow, "Warning", "No trained model available to save.")
                return
            obj_to_save = self.model
        elif button == self.b_SaveCluster:
            if self.cluster is None or self.cluster.empty:
                QMessageBox.warning(self.MainWindow, "Warning", "No trained cluster model available to save.")
                return
            obj_to_save = self.cluster
        else:
            return

        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("Pickle files (*.pkl)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                if FileUtil.saveModel(obj_to_save, file_path):
                    if button == self.b_saveEfficiency:
                        QMessageBox.information(self.MainWindow, "Success", "Model saved successfully.")
                    else:
                        QMessageBox.information(self.MainWindow, "Success", "Cluster model saved successfully.")
                else:
                    QMessageBox.warning(self.MainWindow, "Error", "Failed to save the model/cluster model.")

    def on_error_occurred(self, error_message):
        QMessageBox.warning(self.MainWindow, "Error", error_message)

    def load_model_predict(self):
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilter("Pickle files (*.pkl)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                loaded_object = FileUtil.loadModel(file_path)
                if loaded_object is not None:
                    if isinstance(loaded_object, Pipeline): 
                        self.model = loaded_object
                        QMessageBox.information(self.MainWindow, "Success", "Model loaded successfully.")
                        self.create_model_usage_ui(loaded_object)
                        self.populate_column_list()
                        selected_columns = [col.replace('encoder__', '').replace('scaler__', '') for col in loaded_object[:-1].get_feature_names_out()]
                        for i in range(self.lw_MLData.count()):
                            item = self.lw_MLData.item(i)
                            column_name = item.text()
                            if column_name in self.columns_mapping.keys():
                                db_column_name = self.columns_mapping[column_name]
                                if db_column_name in selected_columns:
                                    item.setCheckState(Qt.CheckState.Checked)
                                else:
                                    item.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        QMessageBox.warning(self.MainWindow, "Warning", "The loaded file is not a regression model. Please load a regression model for predicting efficacy.")
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
            self.figurePrediction.clear()
            self.figurePrediction.set_size_inches(8, 3.8)
            ax = self.figurePrediction.add_subplot(111)
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
            self.canvasPrediction.draw()
        except Exception as e:
            print(e)

    def perform_kmeans_clustering(self):
        try:
            self.connectdb()
            role = self.cb_role.currentText()
            if role == 'Laborer':
                sql = '''SELECT sub_ID, sub_age, sub_health_h, sub_commitment_h, sub_perceptiveness_h, sub_dexterity_h,
                                sub_sociality_h, sup_sub_age_diff, sub_goodness_h, sub_strength_h, sub_openmindedness_h, actual_efficacy_h
                                FROM factory WHERE record_comptype = "Efficacy"'''
                df = self.connector.queryDataset(sql)
                if df is not None:
                    filtered_df = df[['sub_ID', 'sub_age', 'sub_health_h', 'sub_commitment_h', 'sub_perceptiveness_h', 'sub_dexterity_h',
                                    'sub_sociality_h', 'sub_goodness_h', 'sub_strength_h', 'sub_openmindedness_h',
                                    'sup_sub_age_diff', 'actual_efficacy_h']]
                X = filtered_df.drop('actual_efficacy_h', axis=1)
                y = filtered_df['actual_efficacy_h']

                best_features = self.evaluate_feature_importance_for_clustering(X, y)
                print(f"Best features: {best_features}")

                numeric_features = [feature for feature in best_features if pd.api.types.is_numeric_dtype(filtered_df[feature])]
                numeric_features.insert(0, 'actual_efficacy_h')  # Add 'actual_efficacy_h' as the first column

                worker_stats = filtered_df.groupby('sub_ID').agg(
                    **{f'mean_{feature}': (feature, 'mean') for feature in numeric_features}
                ).reset_index()

                data = worker_stats[[f'mean_{feature}' for feature in numeric_features]]

            elif role == "Supervisor":
                sql = '''
                        SELECT sup_ID, sup_age, sup_commitment_h, sup_perceptiveness_h, sup_goodness_h, actual_efficacy_h
                        FROM factory
                        WHERE record_comptype = 'Efficacy'
                        '''
                df = self.connector.queryDataset(sql)
                filtered_df = df[['sup_ID', 'sup_age', 'sup_commitment_h', 'sup_perceptiveness_h', 'sup_goodness_h', 'actual_efficacy_h']]
                X = filtered_df.drop('actual_efficacy_h', axis=1)
                y = filtered_df['actual_efficacy_h']

                best_features = self.evaluate_feature_importance_for_clustering(X, y)
                print(f"Best features: {best_features}")

                numeric_features = [feature for feature in best_features if pd.api.types.is_numeric_dtype(filtered_df[feature])]
                numeric_features.insert(0, 'sup_age')  # Add 'sup_age' as the first column
                numeric_features.insert(1, 'actual_efficacy_h')  # Add 'actual_efficacy_h' as the second column

                supervisor_stats = filtered_df.groupby('sup_ID').agg(
                    **{f'mean_{feature}': (feature, 'mean') for feature in numeric_features}
                ).reset_index()

                data = supervisor_stats[[f'mean_{feature}' for feature in numeric_features]]

            optimal_clusters = self.find_optimal_clusters(data)
            print(f"Optimal number of clusters: {optimal_clusters}")

            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

            data['efficacy_cluster'] = kmeans.fit_predict(data)
            self.cluster = data
            self.visualize_kmeans_clustering()
        except Exception as e:
            QMessageBox.warning(self.MainWindow, "Error", str(e))
            traceback.print_exc()

    def evaluate_feature_importance_for_clustering(self, X, y):
        model = KMeans(n_clusters=3, random_state=0)
        model.fit(X)
        result = permutation_importance(model, X, y, scoring='neg_mean_squared_error', n_jobs=-1)
        sorted_indices = result.importances_mean.argsort()[::-1]
        sorted_features = X.columns[sorted_indices]
        return sorted_features[:3]

    def find_optimal_clusters(self, data, max_clusters=10):
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        optimal_clusters = np.argmin(np.diff(inertias)) + 3
        return optimal_clusters

    def visualize_kmeans_clustering(self):
        try:
            self.role = self.cb_role.currentText()
            self.figureCluster.clear()
            ax = self.figureCluster.add_subplot(111, projection='3d')
            column_names = self.cluster.columns.tolist()
            x_col, y_col, z_col = column_names[:3]
            reverse_mapping = {v: k for k, v in self.columns_mapping.items()}
            x_show = reverse_mapping.get(x_col.replace('mean_', ''), x_col)
            y_show = reverse_mapping.get(y_col.replace('mean_', ''), y_col)
            z_show = reverse_mapping.get(z_col.replace('mean_',''), z_col)
            x = self.cluster[x_col]
            y = self.cluster[y_col]
            z = self.cluster[z_col]
            cluster_labels = self.cluster['efficacy_cluster']
            ax.scatter(x, y, z, c=cluster_labels, cmap='viridis')
            ax.set_xlabel(x_show)
            ax.set_ylabel(y_show)
            ax.set_zlabel(z_show)
            if self.role == "Laborer":
                ax.set_title('Laborer Clustering')
            else:
                ax.set_title('Supervisor Clustering')
            self.canvasCluster.draw()
        except Exception as e:
            print(e)

    def load_cluster_model(self):
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilter("Pickle files (*.pkl)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                loaded_object = FileUtil.loadModel(file_path)
                if loaded_object is not None:
                    if isinstance(loaded_object, pd.DataFrame): 
                        self.cluster = loaded_object
                        QMessageBox.information(self.MainWindow, "Success", "Cluster model loaded successfully.")
                        self.visualize_kmeans_clustering()
                    else:
                        QMessageBox.warning(self.MainWindow, "Warning", "The loaded file is not a clustering model. Please load a KMeans clustering model.")
                else:
                    QMessageBox.warning(self.MainWindow, "Error", "Failed to load the model.")

    def openChangeInfo(self):
        window = QMainWindow()
        self.chartUI = ChangeInformation()
        self.chartUI.setupUi(window)
        self.chartUI.show()

    def logout(self):
        self.MainWindow.close()
        from ui.LoginEx import LoginEx
        window = QMainWindow()
        self.chartUI = LoginEx()
        self.chartUI.setupUi(window)
        self.chartUI.show()