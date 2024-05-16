#MainWindowEx.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow
from ui.MLEx import MLEx
from connector.Connector import Connector
class MainWindowEx(MLEx):
    def __init__(self):
        super().__init__()
        self.connector=Connector()
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        print("setupUi method called")

    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()

    def show(self):
        self.MainWindow.show()


    
    
