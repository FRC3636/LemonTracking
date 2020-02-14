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
        btn.clicked.connect(self.close_function)
        btn.resize(100, 100)
        btn.move(100, 100)
        
        self.show()
    
    def close_function(self):
    	print("yay i did a function thing")
    	sys.exit()
        
        
    
def run():        

    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()

    sys.exit(app.exec_())
    
run()
