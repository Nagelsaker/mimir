#!/usr/bin/env python
import sys
import rospy
from PyQt5.QtWidgets import QApplication
from mimir.Tyr.Gui.OperatorPanel import MainWindow
import qdarkstyle


if __name__ == "__main__":
    rospy.init_node("operator_panel")
    rospy.loginfo("TEST 1")
    app = QApplication(sys.argv)

    darkStylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setStyleSheet(darkStylesheet)
    
    window = MainWindow()
    window.show()
    app.exec()