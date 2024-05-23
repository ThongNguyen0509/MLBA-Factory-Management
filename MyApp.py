#app.py
from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.MainWindowEx import MainWindowEx

app = QApplication([])
mainWindow = QMainWindow()
myWindow = MainWindowEx()
myWindow.setupUi(QMainWindow())
myWindow.show()
app.exec()