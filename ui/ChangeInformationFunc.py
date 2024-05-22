from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox
from ChangeInformation_ui import Ui_MainWindow
from connector.Connector import Connector
import mysql.connector    

class ChangeInformaion(Ui_MainWindow):
    def __init__(self):
        self.connector = Connector()
    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "609618"
        self.connector.connect()
    
    def setupUi(self, ChangeInfWindow):
        super().setupUi(ChangeInfWindow)
        self.ChangeInfWindow = ChangeInfWindow
        self.b_confirmChange.clicked.connect(self.ChangeInformation)
    def setText(self):
        pass
    def ChangeInformation(self):
        try:
            #change variable
            name = self.le_Name
            pw = self.le_password.text()
            #change table and column name
            sql = 'UPDATE account SET Password = %s, Name = %s WHERE UserName = %s'
            self.connector.commitQuery(sql, (pw, name))

        except ValueError:
            dlg = QMessageBox(self.ChangeInfWindow)
            dlg.setWindowTitle("ERROR")
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setText("You have to type correct form of Password to update")
            dlg.exec()