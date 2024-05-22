from ui.ML_ui import Ui_MLWindow
import seaborn as sns
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from connector.Connector import Connector
from sklearn.cluster import KMeans
import numpy as np
class ClusterModel(Ui_MLWindow):
    def __init__(self):
        self.connector = Connector()

    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "609618"
        self.connector.connect()
    def setupUi(self, MLWindow):
        super().setupUi(MLWindow)
        self.MLWindow=MLWindow
        self.b_SeeCluster.clicked.connect(self.ClusterModel)

    def show(self):
        self.MLWindow.show()
    def getDF(self):
        self.connectdb()
        sql = '''
           select * from factorymanagement
            '''
        df = self.connector.queryDataset(sql)
        return df
    def ClusterModel(self):
        df = self.getDF()
        if df is not None:
            filtered_df = df[['sub_ID', 'actual_efficacy_h', 'recorded_efficacy']]

            worker_stats = filtered_df.groupby('sub_ID').agg(
                mean_efficacy=('recorded_efficacy', 'mean'),
                std_efficacy=('recorded_efficacy', 'std')
            ).reset_index()

            kmeans = KMeans(n_clusters=3, random_state=0)
            worker_stats['efficacy_cluster'] = kmeans.fit_predict(worker_stats[['mean_efficacy']])

            stability_threshold = 5
            worker_stats['stability'] = np.where(worker_stats['std_efficacy'] < stability_threshold, 'stable', 'variable')

            self.display_results(worker_stats)
        else:
            print("No data loaded.")

