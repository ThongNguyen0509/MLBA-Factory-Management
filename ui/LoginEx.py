from connector.Connector import Connector
from ui.UI_login_ui import Ui_MainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow, QMessageBox
import bcrypt
from ui.MainWindowEx import MainWindowEx
from constant.constant import Constant
import re

class LoginEx(Ui_MainWindow):
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
        self.to_login_2.clicked.connect(self.toLogin)
        self.to_register_2.clicked.connect(self.toSignUp)
        self.pushButton_2.clicked.connect(self.registerUser)
        self.pushButton_4.clicked.connect(self.loginUser)

    def show(self):
        self.MainWindow.show()

    def toLogin(self):
        self.stackedWidget.setCurrentIndex(1)

    def toSignUp(self):
        self.stackedWidget.setCurrentIndex(0)

    def registerUser(self):
        username = self.lineEdit_6.text()
        name = self.lineEdit.text()
        email = self.lineEdit_2.text()
        password = self.lineEdit_7.text()
        confirm_password = self.lineEdit_8.text()
        agree_terms = self.checkBox_2.isChecked()

        if not all([username, name, email, password, confirm_password]):
            self.showMessage("All fields are required.", QMessageBox.Icon.Warning)
            return

        if not self.validate_username(username):
            self.showMessage("Username should contain only letters, digits, and underscores.", QMessageBox.Icon.Warning)
            return

        if not self.validate_email(email):
            self.showMessage("Invalid email format.", QMessageBox.Icon.Warning)
            return

        if len(password) < 8:
            self.showMessage("Password must be at least 8 characters long.", QMessageBox.Icon.Warning)
            return

        if password != confirm_password:
            self.showMessage("Passwords do not match.", QMessageBox.Icon.Warning)
            return

        if not agree_terms:
            self.showMessage("You must agree with our terms.", QMessageBox.Icon.Warning)
            return

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        self.connectdb()

        if self.connector.conn:
            try:
                self.connector.commitQuery(
                    "INSERT INTO account (username, name, email, password) VALUES (%s, %s, %s, %s)",
                    (username, name, email, hashed_password)
                )
                self.showMessage("Registration successful!", QMessageBox.Icon.Information)
                self.toLogin()
            except Exception as e:
                self.showMessage(f"An error occurred: {e}", QMessageBox.Icon.Critical)
            finally:
                self.connector.disConnect()
        else:
            self.showMessage("Database connection failed.", QMessageBox.Icon.Critical)

    def openMainPage(self):
        window = QMainWindow()
        self.chartUI = MainWindowEx()
        self.chartUI.setupUi(window)
        self.chartUI.show()

    def loginUser(self):
        username = self.lineEdit_9.text()
        password = self.lineEdit_10.text()

        if not all([username, password]):
            self.showMessage("All fields are required.", QMessageBox.Icon.Warning)
            return
        self.connectdb()
        if self.connector.conn:
            try:
                cursor = self.connector.conn.cursor()
                cursor.execute("SELECT id, password, Name FROM account WHERE username = %s", (username,))
                result = cursor.fetchone()
                if result:
                    user_id, stored_password, name = result
                    stored_password = stored_password.encode('utf-8')
                    if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                        self.showMessage("Login successful!", QMessageBox.Icon.Information)
                        self.MainWindow.close()
                        Constant.current_userName = name
                        Constant.current_userID = user_id
                        self.openMainPage()
                    else:
                        self.showMessage("Invalid username or password.", QMessageBox.Icon.Warning)
                else:
                    self.showMessage("Invalid username or password.", QMessageBox.Icon.Warning)
            except Exception as e:
                self.showMessage(f"An error occurred: {e}", QMessageBox.Icon.Critical)
            finally:
                self.connector.disConnect()
        else:
            self.showMessage("Database connection failed.", QMessageBox.Icon.Critical)

    def validate_username(self, username):
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, username))

    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))

    def showMessage(self, message, icon):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(message)
        msg.setWindowTitle("Message")
        msg.exec()