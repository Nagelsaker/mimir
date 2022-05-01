from PyQt5.QtWidgets import  QMainWindow
from PyQt5.QtCore import QSize
from mimir.Tyr.Gui.Ui_MainWindow import Ui_MainWindow
from mimir.Tyr.Gui.SettingsDialog import SettingsDialog
from mimir.Tyr.Hand.HandModel import *
import json
import rospy
import time
import os

STOP = 0
GRIP = 1
UNGRIP = 2
PRECISION = 3
TILT_UP = 4
TILT_DOWN = 5
MOVE_HEIGHT = 6



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.gestureImages = [] # Wrapper
        self.setupUi(self)

        # Setup gesture images
        pathToGraphics = f"{os.path.dirname(os.path.abspath(__file__))}/../graphics/"
        self.gestureImage0.setImage(f"{pathToGraphics}/gesture0.png")
        self.gestureImage0.setActiveImage(f"{pathToGraphics}/gesture0_active.png")
        self.gestureImages.append(self.gestureImage0)
        self.gestureImage1.setImage(f"{pathToGraphics}/gesture1.png")
        self.gestureImage1.setActiveImage(f"{pathToGraphics}/gesture1_active.png")
        self.gestureImages.append(self.gestureImage1)
        self.gestureImage2.setImage(f"{pathToGraphics}/gesture2.png")
        self.gestureImage2.setActiveImage(f"{pathToGraphics}/gesture2_active.png")
        self.gestureImages.append(self.gestureImage2)
        self.gestureImage3.setImage(f"{pathToGraphics}/gesture3.png")
        self.gestureImage3.setActiveImage(f"{pathToGraphics}/gesture3_active.png")
        self.gestureImages.append(self.gestureImage3)
        self.gestureImage4.setImage(f"{pathToGraphics}/gesture4.png")
        self.gestureImage4.setActiveImage(f"{pathToGraphics}/gesture4_active.png")
        self.gestureImages.append(self.gestureImage4)
        self.gestureImage5.setImage(f"{pathToGraphics}/gesture5.png")
        self.gestureImage5.setActiveImage(f"{pathToGraphics}/gesture5_active.png")
        self.gestureImages.append(self.gestureImage5)
        self.gestureImage6.setImage(f"{pathToGraphics}/gesture6.png")
        self.gestureImage6.setActiveImage(f"{pathToGraphics}/gesture6_active.png")
        self.gestureImages.append(self.gestureImage6)
        self.gestureImage7.setImage(f"{pathToGraphics}/ai.png")
        self.gestureImage7.setActiveImage(f"{pathToGraphics}/ai_active.png")
        self.gestureImages.append(self.gestureImage7)
        self.currentGesture = -1

        # Setup Lever Status Image
        self.w = 100 # px
        self.h = 100 # px
        self.setMinimumSize(QSize(self.w, self.h))
        self.leverStatusImg.setImage(f"{pathToGraphics}/gray_btn.png")
        self.leverStatusImg.setActiveImage(f"{pathToGraphics}/green_btn.png")

        self.addAction(self.actionPreferences)
        self.actionPreferences.triggered.connect(self.openDialog)

        # Threshold spin boxes
        self.threshold_wristUp.valueChanged.connect(self.setWristThreshold)
        self.threshold_wristDown.valueChanged.connect(self.setWristThreshold)
        self.threshold_fingerAng1.valueChanged.connect(self.setFingerThreshold)
        self.threshold_thumbAng1.valueChanged.connect(self.setThumbThreshold)
        self.threshold_thumbAng2.valueChanged.connect(self.setThumbThreshold)

        # Buttons
        self.saveData.clicked.connect(self.saveDataPoints)

        # Load settings
        self.advancedUse = rospy.get_param("/mimir/advanced_use")

        if not self.advancedUse:
            self.skeletonWidget.setHidden(True)

        # Load default threshold values
        self.threshold_wristUp.setValue(rospy.get_param("/mimir/threshold_angle_wrist")[0])
        self.threshold_wristDown.setValue(rospy.get_param("/mimir/threshold_angle_wrist")[1])
        self.threshold_fingerAng1.setValue(rospy.get_param("/mimir/threshold_angle_finger"))
        self.threshold_thumbAng1.setValue(rospy.get_param("/mimir/threshold_angle_thumb")[0])
        self.threshold_thumbAng2.setValue(rospy.get_param("/mimir/threshold_angle_thumb")[1])
        
    def setWristThreshold(self):
        th1 = self.threshold_wristUp.value()
        th2 = self.threshold_wristDown.value()
        self.videoStream.th.setWristThreshold([th1, th2])
        
    def setFingerThreshold(self):
        th1 = self.threshold_fingerAng1.value()
        self.videoStream.th.setFingerThreshold(th1)
        
    def setThumbThreshold(self):
        th1 = self.threshold_thumbAng1.value()
        th2 = self.threshold_thumbAng2.value()
        self.videoStream.th.setThumbThreshold([th1, th2])

    def setDepthValue(self, value):
        self.depthDisplay.display(value)
    
    def setMeasuredLeverAngle(self, value):
        self.measuredLeverAngle.display(value)
    
    def setMeasuredLeverPosition(self, value):
        self.measuredLeverPos.display(value)
    
    def setCurrentGoalDisplay(self, goal):
        self.goalDisplay.display(goal)
    
    def setEstLeverAngle(self, value):
        self.estLeverAngle.display(value)
    
    def setEstLeverPos(self, value):
        self.estLeverPos.display(value)

    def openDialog(self):
        settingsDialog = SettingsDialog(self)
        settingsDialog.show()

    def activateGesture(self, idx):
        if idx != self.currentGesture:
            for i in range(len(self.gestureImages)):
                if i == idx:
                    self.currentGesture = idx
                    self.gestureImages[i].activate()
                else:
                    self.gestureImages[i].deactivate()
        if idx == -1: self.currentGesture = idx
    
    def setLeverStatusIcon(self, on):
        '''
        Switch the lever status image to green for t seconds
        '''
        if on:
            self.leverStatusImg.activate()
        else:
            self.leverStatusImg.deactivate()
    
    def updateSkeleton(self, landmarks):
        if self.advancedUse:
            self.skeletonWidget.updatePoints(landmarks)
    
    def saveDataPoints(self):
        if self.advancedUse:
            self.skeletonWidget.save()
        self.videoStream.saveImage()


    def closeEvent(self, event):
        self.videoStream.close()
        event.accept()