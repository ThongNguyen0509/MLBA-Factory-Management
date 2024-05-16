from PyQt6.QtCore import QObject, pyqtSignal
from connector.Connector import Connector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
from multithread.WorkerSignals import WorkerSignals
from PyQt6.QtCore import QRunnable

class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals=WorkerSignals()

    def process_data_for_model(self, connector, selected_columns):
        try:
            sql = "SELECT * FROM factory WHERE record_comptype = 'Efficacy'"
            df = connector.queryDataset(sql)
            X = df[selected_columns]
            y = df['actual_efficacy_h']
            categorical_columns = [col for col in selected_columns if col == 'sub_sex']
            numerical_columns = [col for col in selected_columns if col != 'sub_sex']
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
            pipeline.fit(X, y)
            self.signals.runningSignal.emit(pipeline)
        finally:
            self.signals.finishSignal.emit()

    def visualize_random_forest(self, connector ,pipeline, selected_columns, figure, canvas):
        try:
            sql = "SELECT * FROM factory WHERE record_comptype = 'Efficacy'"
            df = connector.queryDataset(sql)
            X = df[selected_columns]
            y = df['actual_efficacy_h']
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            figure.clear()
            ax = figure.add_subplot(111)
            ax.scatter(y, y_pred, alpha=0.5)
            ax.set_title('Random Forest Regression')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            canvas.draw()
        finally:
            self.finished.emit()