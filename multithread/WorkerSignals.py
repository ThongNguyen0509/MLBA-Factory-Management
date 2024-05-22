from PyQt6.QtCore import QObject, pyqtSignal

class WorkerSignals(QObject):
    runningSignal=pyqtSignal(str,int)
    finishSignal=pyqtSignal()