from connector.Connector import Connector
from ui.UI_login_ui import Ui_MainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow, QMessageBox
import bcrypt

class LoginEx(Ui_MainWindow):
    def __init__(self):
        self.connector = Connector()
    def connectdb(self):
        self.connector.server = "localhost"
        self.connector.port = 3306
        self.connector.database = "factorymanagement"
        self.connector.username = "root"
        self.connector.password = "123456789"
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
                    cursor.execute("SELECT password FROM account WHERE username = %s", (username,))
                    result = cursor.fetchone()

                    if result:
                        stored_password = result[0].encode('utf-8')
                        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                            self.showMessage("Login successful!", QMessageBox.Icon.Information)
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


    def showMessage(self, message, icon):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(message)
        msg.setWindowTitle("Message")
        msg.exec()