#app.py
from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.LoginEx import LoginEx
from ui.MainWindowEx import MainWindowEx
app = QApplication([])
mainWindow = QMainWindow()
myWindow = LoginEx()
myWindow.setupUi(QMainWindow())
myWindow.show() 
app.exec()

1   