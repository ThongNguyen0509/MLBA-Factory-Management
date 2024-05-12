from connector.Connector import Connector
from ui.UI_login_ui import Ui_MainWindow

class LoginEx(Ui_MainWindow):
    def __init__(self):
        self.connector = Connector()
    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "retails"
        self.connector.username = "root"
        self.connector.password = "@Obama123"
        self.connector.connect()
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.to_login_2.clicked.connect(self.toLogin)
        self.to_register_2.clicked.connect(self.toSignUp)

    def show(self):
        self.MainWindow.show()

    def toLogin(self):
        self.stackedWidget.setCurrentIndex(1)

    def toSignUp(self):
        self.stackedWidget.setCurrentIndex(0)