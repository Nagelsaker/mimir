# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1003, 800)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 5, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 3, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 6, 0, 1, 3)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 3, 2, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.gestureImage0 = GestureImage(self.centralwidget)
        self.gestureImage0.setObjectName("gestureImage0")
        self.horizontalLayout_3.addWidget(self.gestureImage0)
        self.gestureImage1 = GestureImage(self.centralwidget)
        self.gestureImage1.setObjectName("gestureImage1")
        self.horizontalLayout_3.addWidget(self.gestureImage1)
        self.gestureImage2 = GestureImage(self.centralwidget)
        self.gestureImage2.setObjectName("gestureImage2")
        self.horizontalLayout_3.addWidget(self.gestureImage2)
        self.gestureImage3 = GestureImage(self.centralwidget)
        self.gestureImage3.setObjectName("gestureImage3")
        self.horizontalLayout_3.addWidget(self.gestureImage3)
        self.gestureImage4 = GestureImage(self.centralwidget)
        self.gestureImage4.setObjectName("gestureImage4")
        self.horizontalLayout_3.addWidget(self.gestureImage4)
        self.gestureImage5 = GestureImage(self.centralwidget)
        self.gestureImage5.setObjectName("gestureImage5")
        self.horizontalLayout_3.addWidget(self.gestureImage5)
        self.gestureImage6 = GestureImage(self.centralwidget)
        self.gestureImage6.setObjectName("gestureImage6")
        self.horizontalLayout_3.addWidget(self.gestureImage6)
        self.gestureImage7 = GestureImage(self.centralwidget)
        self.gestureImage7.setObjectName("gestureImage7")
        self.horizontalLayout_3.addWidget(self.gestureImage7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_2.addWidget(self.line_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.skeletonWidget = SkeletonWidget(self.centralwidget)
        self.skeletonWidget.setObjectName("skeletonWidget")
        self.verticalLayout.addWidget(self.skeletonWidget)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.videoStream = Stream(self.centralwidget)
        self.videoStream.setObjectName("videoStream")
        self.horizontalLayout.addWidget(self.videoStream)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_2.addWidget(self.line_4)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.saveData = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveData.sizePolicy().hasHeightForWidth())
        self.saveData.setSizePolicy(sizePolicy)
        self.saveData.setMinimumSize(QtCore.QSize(110, 0))
        self.saveData.setObjectName("saveData")
        self.horizontalLayout_8.addWidget(self.saveData)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem9)
        self.depthDisplay = QtWidgets.QLCDNumber(self.centralwidget)
        self.depthDisplay.setFrameShape(QtWidgets.QFrame.Box)
        self.depthDisplay.setFrameShadow(QtWidgets.QFrame.Raised)
        self.depthDisplay.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.depthDisplay.setProperty("value", 0.0)
        self.depthDisplay.setObjectName("depthDisplay")
        self.horizontalLayout_6.addWidget(self.depthDisplay)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_6.addWidget(self.label_10)
        spacerItem10 = QtWidgets.QSpacerItem(7, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem10)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_9.addWidget(self.label_11)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem11)
        self.measuredLeverAngle = QtWidgets.QLCDNumber(self.centralwidget)
        self.measuredLeverAngle.setFrameShape(QtWidgets.QFrame.Box)
        self.measuredLeverAngle.setFrameShadow(QtWidgets.QFrame.Raised)
        self.measuredLeverAngle.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.measuredLeverAngle.setProperty("value", 0.0)
        self.measuredLeverAngle.setObjectName("measuredLeverAngle")
        self.horizontalLayout_9.addWidget(self.measuredLeverAngle)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_9.addWidget(self.label_12)
        spacerItem12 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem12)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_10.addWidget(self.label_13)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem13)
        self.estLeverAngle = QtWidgets.QLCDNumber(self.centralwidget)
        self.estLeverAngle.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.estLeverAngle.setObjectName("estLeverAngle")
        self.horizontalLayout_10.addWidget(self.estLeverAngle)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_10.addWidget(self.label_14)
        spacerItem14 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem14)
        self.verticalLayout_4.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_12.addWidget(self.label_17)
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem15)
        self.setGoalSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.setGoalSpinBox.setMinimum(-90.0)
        self.setGoalSpinBox.setMaximum(90.0)
        self.setGoalSpinBox.setObjectName("setGoalSpinBox")
        self.horizontalLayout_12.addWidget(self.setGoalSpinBox)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_12.addWidget(self.label_18)
        spacerItem16 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem16)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_13.addWidget(self.label_19)
        spacerItem17 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem17)
        self.measuredLeverPos = QtWidgets.QLCDNumber(self.centralwidget)
        self.measuredLeverPos.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.measuredLeverPos.setObjectName("measuredLeverPos")
        self.horizontalLayout_13.addWidget(self.measuredLeverPos)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_13.addWidget(self.label_20)
        spacerItem18 = QtWidgets.QSpacerItem(7, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem18)
        self.verticalLayout_4.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_11.addWidget(self.label_15)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem19)
        self.estLeverPos = QtWidgets.QLCDNumber(self.centralwidget)
        self.estLeverPos.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.estLeverPos.setObjectName("estLeverPos")
        self.horizontalLayout_11.addWidget(self.estLeverPos)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_11.addWidget(self.label_16)
        spacerItem20 = QtWidgets.QSpacerItem(7, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem20)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        spacerItem21 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem21)
        self.horizontalLayout_7.addLayout(self.verticalLayout_4)
        spacerItem22 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem22)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem23 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem23)
        self.leverStatusImg = GestureImage(self.centralwidget)
        self.leverStatusImg.setObjectName("leverStatusImg")
        self.verticalLayout_5.addWidget(self.leverStatusImg)
        spacerItem24 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem24)
        self.horizontalLayout_7.addLayout(self.verticalLayout_5)
        spacerItem25 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem25)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem26 = QtWidgets.QSpacerItem(59, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem26)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        spacerItem27 = QtWidgets.QSpacerItem(14, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem27)
        self.threshold_wristUp = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_wristUp.sizePolicy().hasHeightForWidth())
        self.threshold_wristUp.setSizePolicy(sizePolicy)
        self.threshold_wristUp.setMinimum(-180.0)
        self.threshold_wristUp.setMaximum(180.0)
        self.threshold_wristUp.setProperty("value", -15.0)
        self.threshold_wristUp.setObjectName("threshold_wristUp")
        self.horizontalLayout_2.addWidget(self.threshold_wristUp)
        spacerItem28 = QtWidgets.QSpacerItem(33, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem28)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        spacerItem29 = QtWidgets.QSpacerItem(43, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem29)
        self.threshold_wristDown = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_wristDown.sizePolicy().hasHeightForWidth())
        self.threshold_wristDown.setSizePolicy(sizePolicy)
        self.threshold_wristDown.setMinimum(-180.0)
        self.threshold_wristDown.setMaximum(180.0)
        self.threshold_wristDown.setProperty("value", 15.0)
        self.threshold_wristDown.setObjectName("threshold_wristDown")
        self.horizontalLayout_2.addWidget(self.threshold_wristDown)
        spacerItem30 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem30)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        spacerItem31 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem31)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_5.addWidget(self.label_8)
        spacerItem32 = QtWidgets.QSpacerItem(27, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem32)
        self.threshold_thumbAng1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_thumbAng1.sizePolicy().hasHeightForWidth())
        self.threshold_thumbAng1.setSizePolicy(sizePolicy)
        self.threshold_thumbAng1.setMinimum(-180.0)
        self.threshold_thumbAng1.setMaximum(180.0)
        self.threshold_thumbAng1.setProperty("value", -15.0)
        self.threshold_thumbAng1.setObjectName("threshold_thumbAng1")
        self.horizontalLayout_5.addWidget(self.threshold_thumbAng1)
        spacerItem33 = QtWidgets.QSpacerItem(33, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem33)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_5.addWidget(self.label_9)
        spacerItem34 = QtWidgets.QSpacerItem(69, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem34)
        self.threshold_thumbAng2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_thumbAng2.sizePolicy().hasHeightForWidth())
        self.threshold_thumbAng2.setSizePolicy(sizePolicy)
        self.threshold_thumbAng2.setMinimum(-180.0)
        self.threshold_thumbAng2.setMaximum(180.0)
        self.threshold_thumbAng2.setProperty("value", -15.0)
        self.threshold_thumbAng2.setObjectName("threshold_thumbAng2")
        self.horizontalLayout_5.addWidget(self.threshold_thumbAng2)
        spacerItem35 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem35)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        spacerItem36 = QtWidgets.QSpacerItem(46, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem36)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        spacerItem37 = QtWidgets.QSpacerItem(32, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem37)
        self.threshold_fingerAng1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_fingerAng1.sizePolicy().hasHeightForWidth())
        self.threshold_fingerAng1.setSizePolicy(sizePolicy)
        self.threshold_fingerAng1.setMinimum(-180.0)
        self.threshold_fingerAng1.setMaximum(180.0)
        self.threshold_fingerAng1.setProperty("value", 25.0)
        self.threshold_fingerAng1.setObjectName("threshold_fingerAng1")
        self.horizontalLayout_4.addWidget(self.threshold_fingerAng1)
        spacerItem38 = QtWidgets.QSpacerItem(226, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem38)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        spacerItem39 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem39)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.gridLayout.addLayout(self.verticalLayout_2, 3, 1, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 0, 0, 1, 3)
        spacerItem40 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem40, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1003, 22))
        self.menubar.setObjectName("menubar")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionFingers = QtWidgets.QAction(MainWindow)
        self.actionFingers.setObjectName("actionFingers")
        self.actionThumb = QtWidgets.QAction(MainWindow)
        self.actionThumb.setObjectName("actionThumb")
        self.actionWrist = QtWidgets.QAction(MainWindow)
        self.actionWrist.setObjectName("actionWrist")
        self.actionPreferences = QtWidgets.QAction(MainWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.menuSettings.addAction(self.actionPreferences)
        self.menubar.addAction(self.menuSettings.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.saveData.setText(_translate("MainWindow", "Save data"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Depth</span></p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<p>cm</p>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Measured Lever Angle</span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "deg"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Estimated Lever Angle</span></p></body></html>"))
        self.label_14.setText(_translate("MainWindow", "deg"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Goal</span></p></body></html>"))
        self.label_18.setText(_translate("MainWindow", "deg"))
        self.label_19.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Measured Lever Position</span></p></body></html>"))
        self.label_20.setText(_translate("MainWindow", "cm"))
        self.label_15.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Estimated Lever Position</span></p></body></html>"))
        self.label_16.setText(_translate("MainWindow", "cm"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Wrist threshold</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>θ<span style=\" vertical-align:sub;\">w up</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p>θ<span style=\" vertical-align:sub;\">w down</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Thumb threshold</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p>α<span style=\" vertical-align:sub;\">1</span></p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p>α<span style=\" vertical-align:sub;\">2</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Fingers threshold</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p>α</p></body></html>"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.actionFingers.setText(_translate("MainWindow", "Fingers"))
        self.actionThumb.setText(_translate("MainWindow", "Thumb"))
        self.actionWrist.setText(_translate("MainWindow", "Wrist"))
        self.actionPreferences.setText(_translate("MainWindow", "Preferences"))
from Gui.GestureImageWidget import GestureImage
from Gui.SkeletonWidget import SkeletonWidget
from Gui.StreamWidget import Stream
