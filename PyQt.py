import sys
from PyQt5 import QtWidgets

class Window(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowIcon()


sys.exit(app.exec_())
