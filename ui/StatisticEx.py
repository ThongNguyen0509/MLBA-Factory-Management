from connector.Connector import Connector
from ui.SatisticUi_ui import Ui_MainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow, QMessageBox, QTableWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from ui.MLEx import MLEx
import matplotlib.pyplot as plt

class StatisticsEx(Ui_MainWindow):
    def __init__(self):
        self.connector = Connector()
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
        self.b_showPerformance.clicked.connect(self.show_performance_statistics)
        self.b_showEmployee.clicked.connect(self.show_employee_statistics)
        self.setupPlotPerformance()
        self.setupPlotEmployee()
        self.cb_performance.currentTextChanged.connect(self.change_stats_performance)
        self.cb_employee.currentTextChanged.connect(self.change_stats_employee)
        self.b_Logout.clicked.connect(self.logout)
    def show(self):
        self.MainWindow.show()


    def setupPlotPerformance(self):
        self.figurePerformance = plt.figure()
        self.canvasPerformance = FigureCanvas(self.figurePerformance)
        self.toolbarPerformance = NavigationToolbar(self.canvasPerformance, self.MainWindow)
        self.vl_performance.addWidget(self.toolbarPerformance)
        self.vl_performance.addWidget(self.canvasPerformance)

    def setupPlotEmployee(self):
        self.figureEmployee = plt.figure()
        self.canvasEmployee = FigureCanvas(self.figureEmployee)
        self.toolbarEmployee = NavigationToolbar(self.canvasEmployee, self.MainWindow)
        self.vl_Employee.addWidget(self.toolbarEmployee)
        self.vl_Employee.addWidget(self.canvasEmployee)

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

    
    def show_performance_statistics(self):
        if self.cb_performance.currentText() == "Average efficacy of each work group":
            self.show_average_efficacy_workgroup()
        elif self.cb_performance.currentText() == "Top 5 supervisors have most worker resignation":
            self.show_top5_supervisors_has_most_worker_resignation()
        elif self.cb_performance.currentText() == "Average efficacy in groups with below-average and above-average health":
            self.show_average_efficacy_by_health_groups()
        elif self.cb_performance.currentText() == "Average efficacy in groups with below-average and above-average goodness":
            self.show_average_efficacy_by_goodness_groups()

    def show_average_efficacy_workgroup(self):
        self.connectdb()
        sql = f'''
                select sub_workstyle_h as Workgroup, round(avg(actual_efficacy_h),2) as AverageEfficacy from factory
                group by sub_workstyle_h
                order by AverageEfficacy desc
                '''
        self.figurePerformance.set_size_inches(8,2.8)
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statPerformance, df)
        self.figurePerformance.clear()
        ax = self.figurePerformance.add_subplot(111)
        ax.bar(df['Workgroup'], df['AverageEfficacy'])
        ax.set_xlabel('Workgroup')
        ax.set_ylabel('Average Efficacy')
        ax.set_title('Average Efficacy of Each Work Group')
        self.canvasPerformance.draw()

    def show_top5_supervisors_has_most_worker_resignation(self):
        self.connectdb()
        sql = f'''
                SELECT s.sup_lname as SupervisorName, COUNT(s.record_comptype) as ResignationCount
                FROM factory s
                WHERE record_comptype = 'Resignation'
                GROUP BY s.sup_lname
                ORDER BY ResignationCount DESC
                LIMIT 5;
                '''
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statPerformance, df)
        self.figurePerformance.clear()
        self.figurePerformance.set_size_inches(8,2.8)
        ax = self.figurePerformance.add_subplot(111)
        ax.bar(df['SupervisorName'], df['ResignationCount']) 
        ax.set_xlabel('Supervisor')
        ax.set_ylabel('Resignation Count')
        ax.set_title('Top 5 Supervisors Has Most Worker Resignation')
        self.canvasPerformance.draw()

    def show_average_efficacy_by_health_groups(self):
        self.connectdb()
        sql = """
            SELECT
                CASE
                    WHEN sub_health_h < (SELECT AVG(sub_health_h) FROM factory) THEN 'Below Average Health'
                    ELSE 'Above Average Health'
                END AS HealthGroup,
                ROUND(AVG(recorded_efficacy), 2) AS AverageEfficacy
            FROM
                factory
            WHERE record_comptype = "Efficacy"
            GROUP BY
                HealthGroup;
        """
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statPerformance, df)
        self.figurePerformance.clear()
        self.figurePerformance.set_size_inches(8,2.8)
        ax = self.figurePerformance.add_subplot(111)
        ax.bar(df['HealthGroup'], df['AverageEfficacy'])
        ax.set_xlabel('Health Group')
        ax.set_ylabel('Average Efficacy')
        ax.set_title('Average Efficacy in Groups with Below-Average and Above-Average Health')
        self.canvasPerformance.draw()

    def show_average_efficacy_by_goodness_groups(self):
        self.connectdb()
        sql = """
            SELECT
                CASE
                    WHEN sub_health_h < (SELECT AVG(sub_goodness_h) FROM factory) THEN 'Below Average Goodness'
                    ELSE 'Above Average Goodness'
                END AS GoodnessGroup,
                ROUND(AVG(recorded_efficacy), 2) AS AverageEfficacy
            FROM
                factory
            WHERE record_comptype = "Efficacy"
            GROUP BY
                GoodnessGroup;
        """
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statPerformance, df)
        self.figurePerformance.clear()
        self.figurePerformance.set_size_inches(8,2.8)
        ax = self.figurePerformance.add_subplot(111)
        ax.bar(df['GoodnessGroup'], df['AverageEfficacy'])
        ax.set_xlabel('Goodness Group')
        ax.set_ylabel('Average Efficacy')
        ax.set_title('Average Efficacy in Groups with Below-Average and Above-Average Goodness')
        self.canvasPerformance.draw()

    def show_employee_statistics(self):
        if self.cb_employee.currentText() == "Distribution of gender":
            self.show_gender_distribution()
        elif self.cb_employee.currentText() == "Distribution of age":
           self.show_age_distribution()
        elif self.cb_employee.currentText() == "Average worker in shifts by gender":
            self.average_worker_in_shifts_by_gender()
        elif self.cb_employee.currentText() == "Top 5 resignation reasons":
            self.top5_resignations_reason()
        else:
            print('as')
        

    def show_gender_distribution(self):
        self.connectdb()
        sql = "SELECT sub_sex AS Gender, COUNT(*) AS Count FROM factory WHERE record_comptype = 'Efficacy' GROUP BY sub_sex"
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statEmployee, df)
        self.figureEmployee.clear()
        self.figurePerformance.set_size_inches(8,2.3)
        ax = self.figureEmployee.add_subplot(111)
        ax.bar(df['Gender'], df['Count'])
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Gender')
        self.canvasEmployee.draw()

    def show_age_distribution(self):
        self.connectdb()
        sql = "SELECT sub_age AS Age, COUNT(*) AS Count FROM factory WHERE record_comptype = 'Efficacy' GROUP BY sub_age"
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statEmployee, df)
        self.figureEmployee.clear()
        self.figurePerformance.set_size_inches(8,2.3)
        ax = self.figureEmployee.add_subplot(111)
        ax.bar(df['Age'], df['Count'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Age')
        self.canvasEmployee.draw()

    def average_worker_in_shifts_by_gender(self):
        self.connectdb()
        sql = "SELECT sub_sex AS Gender, sub_shift AS Shift, COUNT(*) AS Count FROM factory WHERE record_comptype = 'Efficacy' GROUP BY sub_sex, sub_shift"
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statEmployee, df)
        self.figureEmployee.clear()
        self.figurePerformance.set_size_inches(8,2.3)
        ax = self.figureEmployee.add_subplot(111)
        pivot_table = df.pivot_table(index='Gender', columns='Shift', values='Count', aggfunc='sum')
        pivot_table.plot(kind='bar', ax=ax, rot=0)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Average Worker in Shifts by Gender')
        ax.legend(title='Shift')

        self.canvasEmployee.draw()

    def top5_resignations_reason(self):
        self.connectdb()
        sql = "SELECT behav_cause_h AS ResignReason, COUNT(*) AS Count FROM factory WHERE behav_cause_h IS NOT NULL GROUP BY behav_cause_h ORDER BY Count DESC LIMIT 5"
        df = self.connector.queryDataset(sql)
        self.showDataIntoTableWidget(self.tw_statEmployee, df)
        self.figureEmployee.clear()
        self.figurePerformance.set_size_inches(8,2.3)
        ax = self.figureEmployee.add_subplot(111)
        ax.bar(df['ResignReason'], df['Count'])
        ax.set_xlabel('Resignation Reason')
        ax.set_ylabel('Count')
        ax.set_title('Top 5 Resignation Reasons')
        self.canvasEmployee.draw()

    def change_stats_performance(self):
        self.figurePerformance.clear()
        self.canvasPerformance.draw() 
        self.tw_statPerformance.setRowCount(0)
        self.tw_statPerformance.setColumnCount(0)
    
    def change_stats_employee(self):
        self.figureEmployee.clear()
        self.canvasEmployee.draw() 
        self.tw_statEmployee.setRowCount(0)
        self.tw_statEmployee.setColumnCount(0)

    def logout(self):
        self.MainWindow.close()
        from ui.LoginEx import LoginEx
        window = QMainWindow()
        self.chartUI = LoginEx()
        self.chartUI.setupUi(window)
        self.chartUI.show()