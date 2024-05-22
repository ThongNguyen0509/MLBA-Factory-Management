from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.ChangeInformationFunc import ChangeInformaion

app = QApplication([])
mainWindow = QMainWindow()
myWindow = ChangeInformaion()
myWindow.setupUi(QMainWindow())
myWindow.show()
app.exec()

