#Day-29  covid-19 Detction using Deep Learning
#From implementation generated from reading ui file
# WARNING!  All changes made in this file will be lost

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import Sequential

#initialize nn
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Convert pooling features space to large feature vector for fully conectted layer

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout

import os
import cv2
#################
from imutils import contours
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.neutral_network import MLPClassifer
import csv
import scipy
import scipy.io as sio
import imutils
import os
import mahotas as mt

############

p=1;

class Ui_MainWindow(object):
    def setupUi(self,MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resiae(800,600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage=QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRec(160,370,151,51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLb1 = QtWidgets.QLabel(self.centralwidget)
        self.imageLb1.Geometry(QtCore.QRect(200,30,361,261))
        self.imageLb1.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLb1.setText("")
        self.imageLb1.setObjectName("imageLb1")
        self.label_2 = QtWidgets.QLabel(self.central widget)
        self.label_2.setGeometry(QtCore.QRect(110,20,621,20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160,450,151,51))
        self.Classify.setObjectName("Classify")
        self.label = QTWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430,370,111,16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QLabel(self.centralwidget)
        self.Training.setGeometry(QtCore.QREct(400,390,211,51))
        self.textEdit.setObjectName("textEdit")
        mainWindow.SetCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0,0,800,26))
        self.menubar.setObjectName('statusbar')
        MainWindow.setStatusBAr(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.tainingFunciton)

    def retranslateUi(self,MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow","MainWindow"))
        self.BrowseImage.setTest(_translate("mainWindow","Browse Image"))
        self.label_2.setTextt(_translate("MainWindow","              Covid-19"))
        self.Classify.setText(_translate("MainWindow","Classify"))
        self.label.setText(_translate("MainWindow","Recognized Class"))
        self.Training.setText(_translate("MainWindow","Training"))

    def LoadImage(self):
        fileName = ZtWidgets.QFileDialog.getopenFileName(None,"select Training")
        if FileName: #If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName)#Setup pixamp with the provided
            pixmap = pipxmap.scaled(self.imageLbl.width(),self.imageLbl.height())
            self.imageLbl.setPixmap(pixmap) #Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        json_file = open('model.json','r')
        loaded_model_json = json_file.read()
        #To be Continued
                                    






































        
        
        


































        





































































