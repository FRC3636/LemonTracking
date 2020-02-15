import sys
from PyQt5 import QtWidgets, QtCore
import PyQt_Window as Qt
        
    
def run():        

    app = QtWidgets.QApplication(sys.argv)
    GUI = Qt.Window()

    sys.exit(app.exec_())

run()
