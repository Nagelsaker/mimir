import time
import json
import cv2
import rospy
from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from mimir.Tyr.FSM import FSM
from mimir.Tyr.Utility.utils import generateFilename


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    activateGesture = pyqtSignal(int)
    updateSkeleton = pyqtSignal(list)
    setDepthValue = pyqtSignal(float)
    setMeasuredLeverAngle = pyqtSignal(float)
    setMeasuredLeverPosition = pyqtSignal(float)
    setEstLeverAngle = pyqtSignal(float)
    setEstLeverPos = pyqtSignal(float)
    setLeverStatusIcon = pyqtSignal(bool)

    def __init__(self, parent, w, h):
        super().__init__(parent)
        self.w = w
        self.h = h
        self.fsm = FSM()

    def run(self):
        self.fsm.run(self)
    
    def setWristThreshold(self, threshold):
        self.fsm.setWristThreshold(threshold)
    
    def setFingerThreshold(self, threshold):
        self.fsm.setFingerThreshold(threshold)
    
    def setThumbThreshold(self, threshold):
        self.fsm.setThumbThreshold(threshold)
    
    def setCurrentGoal(self, goal):
        self.fsm.setCurrentGoal(goal)
    
    def getCurrentImage(self):
        return self.fsm.getCurrentImage()

class Stream(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pathToDataset = rospy.get_param("/mimir/path_to_dataset")

        # create a label
        self.label = QLabel(self)
        scale = 0.4
        self.w = int(1920*scale)
        self.h = int(1080*scale)
        self.setMinimumSize(QSize(self.w, self.h))
        # self.label.move(280, 120)
        self.label.resize(self.w, self.h)

        self.th = Thread(self, self.w-5, self.h-5)
        self.th.changePixmap.connect(self.setImage)
        self.th.activateGesture.connect(self.parent().parent().activateGesture)
        self.th.updateSkeleton.connect(self.parent().parent().updateSkeleton)
        self.th.setDepthValue.connect(self.parent().parent().setDepthValue)
        self.th.setMeasuredLeverAngle.connect(self.parent().parent().setMeasuredLeverAngle)
        self.th.setMeasuredLeverPosition.connect(self.parent().parent().setMeasuredLeverPosition)
        self.th.setEstLeverAngle.connect(self.parent().parent().setEstLeverAngle)
        self.th.setEstLeverPos.connect(self.parent().parent().setEstLeverPos)
        self.th.setLeverStatusIcon.connect(self.parent().parent().setLeverStatusIcon)
        self.th.start()

    def setSize(self, width, height):
        self.label.resize(width, height)
        self.w = width
        self.h = height
    
    def close(self):
        self.th.requestInterruption()
        time.sleep(0.2)
        while self.th.isRunning():
            continue
    
    def saveImage(self):
        fname = generateFilename(self.pathToDataset, "jpg", "workspace")
        im = self.th.getCurrentImage()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fname, im)



    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))