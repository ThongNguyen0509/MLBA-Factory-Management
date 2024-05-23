from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox
from ui.ChangeInformationUI_ui import Ui_MainWindow
from connector.Connector import Connector
from constant.constant import Constant
import bcrypt

class ChangeInformation(Ui_MainWindow):
    def __init__(self):
        self.connector = Connector()
        self.constant = Constant()
        self.current_id = self.constant.current_userID

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
        self.b_confirmChange.clicked.connect(self.changeInformation)
        self.b_cancel.clicked.connect(self.cancel)
        self.setInformation()

    def show(self):
        self.MainWindow.show()

    def setInformation(self):
        self.connectdb()
        cursor = self.connector.conn.cursor()
        cursor.execute("SELECT name, password, UserName, Email FROM account WHERE id = %s", (self.current_id,))
        result = cursor.fetchone()

        if result:
            name, password, username, email = result
            self.le_Name.setText(name)
            self.le_password.setText(password)
            self.le_confirmPass.setText(password)
            self.le_email.setText(email)
            self.le_userName.setText(username)

        self.connector.disConnect()

    def changeInformation(self):
        name = self.le_Name.text()
        password = self.le_password.text()
        confirm_password = self.le_confirmPass.text()

        if not all([name, password, confirm_password]):
            self.showMessage("All fields are required.", QMessageBox.Icon.Warning)
            return

        if password != confirm_password:
            self.showMessage("Passwords do not match.", QMessageBox.Icon.Warning)
            return

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        self.connectdb()

        if self.connector.conn:
            try:
                cursor = self.connector.conn.cursor()
                cursor.execute("UPDATE account SET name = %s, password = %s WHERE id = %s",
                                (name, hashed_password, self.current_id))
                self.connector.conn.commit()
                self.showMessage("Information updated successfully!", QMessageBox.Icon.Information)
            except Exception as e:
                self.showMessage(f"An error occurred: {e}", QMessageBox.Icon.Critical)
            finally:
                self.connector.disConnect()

    def showMessage(self, message, icon):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(message)
        msg.setWindowTitle("Message")
        msg.exec()
    
    def cancel(self):
        self.MainWindow.close()