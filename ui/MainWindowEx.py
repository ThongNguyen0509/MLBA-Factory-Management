import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow
from ui.ClusterModel import Ui_MLWindow
from connector.Connector import Connector
class MainWindowEx(Ui_MLWindow):
    def __init__(self):
        super().__init__()
        self.connector=Connector()
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow

        
    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "retails"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()

    def show(self):
        self.MainWindow.show()


    
    
