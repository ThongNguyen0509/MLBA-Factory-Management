#MainWindowEx.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow
from ui.MainPage_ui import Ui_MainWindow
from connector.Connector import Connector
from ui.MLEx import MLEx
from ui.StatisticEx import StatisticsEx
from ui.ChangeInformationEx import ChangeInformation
from constant.constant import Constant

class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.connector=Connector()
        
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.l_setName.setText(f"Welcome, {Constant.current_userName}")
        self.pushButton_Models.clicked.connect(self.openModels)
        self.pushButton_Statistic.clicked.connect(self.openStatistics)
        self.pushButton_User.clicked.connect(self.openChangeInfo)
        self.b_Logout_2.clicked.connect(self.logout)
        self.show_employee_information()
        self.comboBox_Role.currentTextChanged.connect(self.show_employee_information)

    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()

    def show(self):
        self.MainWindow.show()

    def openModels(self):
        self.MainWindow.close()
        window = QMainWindow()
        self.chartUI = MLEx()
        self.chartUI.setupUi(window)
        self.chartUI.show() 

    def openStatistics(self):
        self.MainWindow.close()
        window = QMainWindow()
        self.chartUI = StatisticsEx()
        self.chartUI.setupUi(window)
        self.chartUI.show()         

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

    def showDataIntoTableWidget(self, table, df):
        table.setRowCount(0)
        table.setColumnCount(len(df.columns))
        for i in range(len(df.columns)):
            columnHeader = df.columns[i]
            table.setHorizontalHeaderItem(i, QTableWidgetItem(columnHeader))
        row=0
        for item in df.iloc:
            arr = item.values.tolist()
            table.insertRow(row)
            j=0
            for data in arr:
                table.setItem(row, j, QTableWidgetItem(str(data)))
                j +=1
            row +=1
        table.resizeColumnsToContents()

    def show_employee_information(self):
        self.tw_empInfo.setRowCount(0)
        self.tw_empInfo.setColumnCount(0)
        self.connectdb()
        if self.comboBox_Role.currentText() == 'Labor':
            sql = '''
                select distinct sub_ID as WorkerID, sub_lname as Name, sub_sex as Gender, sub_age as Age, sub_role as Position from factory where sub_role = 'Laborer'
                '''
            df = self.connector.queryDataset(sql)
            self.showDataIntoTableWidget(self.tw_empInfo, df)
        if self.comboBox_Role.currentText() == 'Supervisor':
            sql = '''
                select distinct sup_ID as SupervisorID, sup_lname as Name, sup_age as Age, sup_sex as Gender, sup_role as Position from factory where sup_role = 'Shift Manager' or sup_role = 'Team Leader' or sup_role = 'Production Director'
                '''            
            df = self.connector.queryDataset(sql)
            self.showDataIntoTableWidget(self.tw_empInfo, df)


