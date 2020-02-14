import sys
from PyQt5 import QtWidgets, QtCore

class Window(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("PyQt testing")
        self.home()
        
    def home(self):
        btn = QtWidgets.QPushButton("Quit", self)
        btn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        btn.resize(100, 100)
        btn.move(100, 100)
        
        self.show()
        
        
    
def run():        

    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()

    sys.exit(app.exec_())
    
run()
