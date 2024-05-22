# Form implementation generated from reading ui file 'd:\UEL\HK6\ML in BA\MLBA-Factory-Management\ui\ML.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MLWindow(object):
    def setupUi(self, MLWindow):
        MLWindow.setObjectName("MLWindow")
        MLWindow.resize(1120, 680)
        MLWindow.setIconSize(QtCore.QSize(30, 30))
        self.centralwidget = QtWidgets.QWidget(parent=MLWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 70, 1081, 541))
        self.tabWidget.setMouseTracking(True)
        self.tabWidget.setTabletTracking(True)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(20, 20))
        self.tabWidget.setElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.Cluster = QtWidgets.QWidget()
        self.Cluster.setObjectName("Cluster")
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.Cluster)
        self.groupBox_4.setGeometry(QtCore.QRect(30, 20, 671, 101))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_4.setObjectName("groupBox_4")
        self.cb_Role = QtWidgets.QComboBox(parent=self.groupBox_4)
        self.cb_Role.setGeometry(QtCore.QRect(20, 40, 111, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.cb_Role.setFont(font)
        self.cb_Role.setStyleSheet("background-color: rgb(226, 244, 255);")
        self.cb_Role.setObjectName("cb_Role")
        self.cb_Role.addItem("")
        self.cb_Role.addItem("")
        self.label_14 = QtWidgets.QLabel(parent=self.groupBox_4)
        self.label_14.setGeometry(QtCore.QRect(20, 20, 71, 16))
        self.label_14.setObjectName("label_14")
        self.b_SeeCluster = QtWidgets.QPushButton(parent=self.groupBox_4)
        self.b_SeeCluster.setGeometry(QtCore.QRect(480, 70, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_SeeCluster.setFont(font)
        self.b_SeeCluster.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_SeeCluster.setObjectName("b_SeeCluster")
        self.b_SaveCluster = QtWidgets.QPushButton(parent=self.groupBox_4)
        self.b_SaveCluster.setGeometry(QtCore.QRect(260, 70, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_SaveCluster.setFont(font)
        self.b_SaveCluster.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_SaveCluster.setObjectName("b_SaveCluster")
        self.b_LoadCluster = QtWidgets.QPushButton(parent=self.groupBox_4)
        self.b_LoadCluster.setGeometry(QtCore.QRect(370, 70, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_LoadCluster.setFont(font)
        self.b_LoadCluster.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_LoadCluster.setObjectName("b_LoadCluster")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(parent=self.Cluster)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(30, 140, 841, 371))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_employee.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.tabWidget.addTab(self.Cluster, icon, "")
        self.Efficiency = QtWidgets.QWidget()
        self.Efficiency.setObjectName("Efficiency")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.Efficiency)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 10, 711, 151))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_10 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(20, 20, 111, 16))
        self.label_10.setObjectName("label_10")
        self.b_trainEfficiency = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.b_trainEfficiency.setGeometry(QtCore.QRect(490, 110, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_trainEfficiency.setFont(font)
        self.b_trainEfficiency.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_trainEfficiency.setObjectName("b_trainEfficiency")
        self.listWidget = QtWidgets.QListWidget(parent=self.groupBox_2)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 256, 91))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)
        self.b_saveEfficiency = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.b_saveEfficiency.setGeometry(QtCore.QRect(290, 110, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_saveEfficiency.setFont(font)
        self.b_saveEfficiency.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_saveEfficiency.setObjectName("b_saveEfficiency")
        self.b_loadEfficiency = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.b_loadEfficiency.setGeometry(QtCore.QRect(390, 110, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_loadEfficiency.setFont(font)
        self.b_loadEfficiency.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_loadEfficiency.setObjectName("b_loadEfficiency")
        self.b_visualizeEfficiency = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.b_visualizeEfficiency.setGeometry(QtCore.QRect(590, 110, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.b_visualizeEfficiency.setFont(font)
        self.b_visualizeEfficiency.setStyleSheet("background-color:rgb(226, 244, 255);\n"
"color: rgb(0, 0, 0);\n"
"")
        self.b_visualizeEfficiency.setObjectName("b_visualizeEfficiency")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.Efficiency)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 170, 791, 341))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gb_modelUsage = QtWidgets.QGroupBox(parent=self.Efficiency)
        self.gb_modelUsage.setGeometry(QtCore.QRect(830, 10, 201, 281))
        self.gb_modelUsage.setObjectName("gb_modelUsage")
        self.formLayoutWidget = QtWidgets.QWidget(parent=self.gb_modelUsage)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 30, 160, 231))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_efficiency.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.tabWidget.addTab(self.Efficiency, icon1, "")
        self.ChurnTab = QtWidgets.QWidget()
        self.ChurnTab.setObjectName("ChurnTab")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.ChurnTab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(60, 120, 681, 321))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_8 = QtWidgets.QLabel(parent=self.ChurnTab)
        self.label_8.setGeometry(QtCore.QRect(10, 20, 31, 31))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_admin.png"))
        self.label_8.setScaledContents(True)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.ChurnTab)
        self.label_9.setGeometry(QtCore.QRect(50, 20, 101, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_churn.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.tabWidget.addTab(self.ChurnTab, icon2, "")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, -10, 91, 81))
        self.label_2.setMaximumSize(QtCore.QSize(91, 81))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_logo.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(800, 10, 31, 31))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_language.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(840, 10, 41, 31))
        self.label_4.setObjectName("label_4")
        self.line = QtWidgets.QFrame(parent=self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 50, 1091, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.line.setFont(font)
        self.line.setStyleSheet("color: rgb(0, 0, 127);\n"
"border-color: rgb(0, 0, 127);")
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.label_6 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(110, 10, 31, 31))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("d:\\UEL\\HK6\\ML in BA\\MLBA-Factory-Management\\ui\\../image/img_admin.png"))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(150, 10, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.b_changeProfile = QtWidgets.QPushButton(parent=self.centralwidget)
        self.b_changeProfile.setGeometry(QtCore.QRect(900, 10, 101, 31))
        self.b_changeProfile.setObjectName("b_changeProfile")
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1020, 10, 75, 31))
        self.pushButton.setObjectName("pushButton")
        MLWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MLWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1120, 18))
        self.menubar.setObjectName("menubar")
        MLWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MLWindow)
        self.statusbar.setObjectName("statusbar")
        MLWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MLWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MLWindow)

    def retranslateUi(self, MLWindow):
        _translate = QtCore.QCoreApplication.translate
        MLWindow.setWindowTitle(_translate("MLWindow", "MainWindow"))
        self.tabWidget.setToolTip(_translate("MLWindow", "<html><head/><body><p>Machine Learning</p></body></html>"))
        self.groupBox_4.setTitle(_translate("MLWindow", "Employee\'s Information"))
        self.cb_Role.setItemText(0, _translate("MLWindow", "Laborer"))
        self.cb_Role.setItemText(1, _translate("MLWindow", "Leam Leader"))
        self.label_14.setToolTip(_translate("MLWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label_14.setText(_translate("MLWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#00007f;\">Role</span></p></body></html>"))
        self.b_SeeCluster.setText(_translate("MLWindow", "See Cluster analysis model"))
        self.b_SaveCluster.setText(_translate("MLWindow", "Save model"))
        self.b_LoadCluster.setText(_translate("MLWindow", "Load model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Cluster), _translate("MLWindow", "Cluster Analysis"))
        self.groupBox_2.setTitle(_translate("MLWindow", "Independent Variables"))
        self.label_10.setToolTip(_translate("MLWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label_10.setText(_translate("MLWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#00007f;\">Choose Variables</span></p></body></html>"))
        self.b_trainEfficiency.setText(_translate("MLWindow", "Train model"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MLWindow", "Dexterxity"))
        item = self.listWidget.item(1)
        item.setText(_translate("MLWindow", "Worker Supervisor Age Diff"))
        item = self.listWidget.item(2)
        item.setText(_translate("MLWindow", "Sociality"))
        item = self.listWidget.item(3)
        item.setText(_translate("MLWindow", "Goodness"))
        item = self.listWidget.item(4)
        item.setText(_translate("MLWindow", "Strength"))
        item = self.listWidget.item(5)
        item.setText(_translate("MLWindow", "Open-minded"))
        item = self.listWidget.item(6)
        item.setText(_translate("MLWindow", "Health"))
        item = self.listWidget.item(7)
        item.setText(_translate("MLWindow", "Commitment"))
        item = self.listWidget.item(8)
        item.setText(_translate("MLWindow", "Perceptiveness"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.b_saveEfficiency.setText(_translate("MLWindow", "Save model"))
        self.b_loadEfficiency.setText(_translate("MLWindow", "Load model"))
        self.b_visualizeEfficiency.setText(_translate("MLWindow", "Visualize model"))
        self.gb_modelUsage.setTitle(_translate("MLWindow", "Model Usage"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Efficiency), _translate("MLWindow", "Efficiency Prediction"))
        self.label_9.setText(_translate("MLWindow", "Welcome, Robert"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ChurnTab), _translate("MLWindow", "Churn Prediction"))
        self.label_4.setText(_translate("MLWindow", "English"))
        self.label_7.setText(_translate("MLWindow", "Welcome, Robert"))
        self.b_changeProfile.setText(_translate("MLWindow", "Change Profile"))
        self.pushButton.setText(_translate("MLWindow", "Log Out"))
