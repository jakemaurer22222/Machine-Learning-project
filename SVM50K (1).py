import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import random
from sklearn import svm
import os
import subprocess
import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:34:22 2019

@author: shack
"""
""""""


class svmClass:
    def __init__(self):
        dataFile = open("adult.txt", "r")
        errorNum = 0
        self.X = []
        self.Y = []
        self.XY = []
        self.XYvalid = []
        self.numPoints = 0
        for i in dataFile:
            if (errorNum < 40701):
                #print(errorNum)
                self.numPoints = self.numPoints + 1
                x1 = 0
                x2 = 0
                x3 = 0
                x4 = 0
                x5 = 0
                x6 = 0
                x7 = 0
                x8 = 0
                x9 = 0
                x10 = 0
                x11 = 0
                x12 = 0
                x13 = 0
                x14 = 0

                # age,workclass,weight,education, educationNum, marital, occupation, relationship,
                # race, sex, capitalGain, captialLoss, hours, country, 50class = i.split(none,14)
                errorNum = errorNum + 1
                currentLine = i.split();
                #(errorNum)
                age = currentLine[0]
                workclass = currentLine[1]
                weight = currentLine[2]
                education = currentLine[3]
                educationNum = currentLine[4]
                marital = currentLine[5]
                occupation = currentLine[6]
                relationship = currentLine[7]
                race = currentLine[8]
                sex = currentLine[9]
                capitalGain = currentLine[10]
                capitalLoss = currentLine[11]
                hours = currentLine[12]
                country = currentLine[13]
                classif = currentLine[14]
                x1 = 1
                x2 = 1
                x3 = 1
                x4 = 1
                x5 = 1
                x6 = 1
                x7 = 1
                x8 = 1
                x9 = 1
                x10 = 1
                x11 = 1
                x12 = 1
                x13 = 1
                x14 = 1
                x15 = 1
                x1 = age
                # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
                if (workclass == "Private"):
                    x2 = 42000
                if (workclass == "Self-emp-not-inc"):
                    x2 = 36000
                if (workclass == "Self-emp-inc"):
                    x2 = 45000
                if (workclass == "Federal-gov"):
                    x2 = 84153
                if (workclass == "Local-gov"):
                    x2 = 47230
                if (workclass == "State-gov"):
                    x2 = 53180
                if (workclass == "Without-pay"):
                    x2 = 10000
                if (workclass == "Never-worked"):
                    x2 = 0
                x3 = weight
                # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
                if (education == "Bachelors"):
                    x4 = 50556
                if (education == "Some-college"):
                    x4 = 319000
                if (education == "11th"):
                    x4 = 20241
                if (education == "HS-grad"):
                    x4 = 30500
                if (education == "Prof-school"):
                    x4 = 42000
                if (education == "Assoc-acdm"):
                    x4 = 39000
                if (education == "Assoc-voc"):
                    x4 = 39000
                if (education == "9th"):
                    x4 = 22717
                if (education == "7th-8th"):
                    x4 = 17962
                if (education == "12th"):
                    x4 = 23717  # Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
                if (education == "Masters"):
                    x4 = 79818
                if (education == "1st-4th"):
                    x4 = 12000
                if (education == "10th"):
                    x4 = 23717
                if (education == "Doctorate"):
                    x4 = 84396
                if (education == "5th-6th"):
                    x4 = 14000
                if (education == "Preschool"):
                    x4 = 10000
                x5 = educationNum
                #: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
                if (marital == "Married-civ-spouse"):
                    x6 = 48150
                if (marital == "Divorced"):
                    x6 = 45000
                if (marital == "Never-married"):
                    x6 = 40000
                if (marital == "Separated"):
                    x6 = 42000
                if (marital == "Widowed"):
                    x6 = 46000
                if (marital == "Married-spouse-absent"):
                    x6 = 465000
                if (marital == "Married-AF-spouse"):
                    x6 = 49000
                # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,
                # Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical,
                # Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
                # Armed-Forces
                if (occupation == "Tech-support"):
                    x7 = 59807
                if (occupation == "Craft-repair"):
                    x7 = 43490
                if (occupation == "Other-service"):
                    x7 = 40000
                if (occupation == "Sales"):
                    x7 = 67960
                if (occupation == "Exec-managerial"):
                    x7 = 85254
                if (occupation == "Prof-specialty"):
                    x7 = 217000
                if (occupation == "Handlers-cleaners"):
                    x7 = 30000
                if (occupation == "Machine-op-inspct"):
                    x7 = 47000
                if (occupation == "Adm-clerical"):
                    x7 = 75000
                if (occupation == "Farming-fishing"):
                    x7 = 42360
                if (occupation == "Transport-moving"):
                    x7 = 26000
                if (occupation == "Priv-house-serv"):
                    x7 = 24165
                if (occupation == "Protective-serv"):
                    x7 = 55000
                if (occupation == "Armed-Forces"):
                    x7 = 44195
                # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
                if (relationship == "Wife"):
                    x8 = 1
                if (relationship == "Own-child"):
                    x8 = 2
                if (relationship == "Husband"):
                    x8 = 3
                if (relationship == "Not-in-family"):
                    x8 = 4
                if (relationship == "Other-relative"):
                    x8 = 5
                if (relationship == "Unmarried"):
                    x8 = 6
                # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
                if (race == "White"):
                    x9 = 1
                if (race == "Asian-Pac-Islander"):
                    x9 = 2
                if (race == "Amer-Indian-Eskimo"):
                    x9 = 3
                if (race == "Other"):
                    x9 = 4
                if (race == "Black"):
                    x9 = 5
                # Male, Female
                if (sex == "Male"):
                    x10 = 1
                if (sex == "Female"):
                    x10 = 0
                s = 0
                if (classif == "<=50K"):
                    s = -1
                if (classif == ">50K"):
                    s = 1
                x11 = capitalGain
                x12 = capitalLoss
                x13 = hours
                # United-States, Cambodia, England, Puerto-Rico, Canada,
                # Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece,
                # South, China, Cuba,
                # Iran, Honduras, Philippines, Italy, Poland,
                # Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
                # Dominican-Republic, Laos, Ecuador, Taiwan, Haiti,
                # Columbia, Hungary, Guatemala, Nicaragua, Scotland,
                # Thailand, Yugoslavia, El-Salvador,
                # Trinadad&Tobago, Peru, Hong, Holand-Netherlands
                if (country == "United-States"):
                    x14 = 59160
                if (country == "Cambodia"):
                    x14 = 5
                if (country == "England"):
                    x14 = 3
                if (country == "Puerto-Rico"):
                    x14 = 4
                if (country == "Canada"):
                    x14 = 42790
                if (country == "Germany"):
                    x14 = 43700
                if (country == "Outlying-US(Guam-USVI-etc)"):
                    x14 = 7
                if (country == "India"):
                    x14 = 1790
                if (country == "Japan"):
                    x14 = 38520
                if (country == "Greece"):
                    x14 = 18340
                if (country == "South"):
                    x14 = 28380
                if (country == "China"):
                    x14 = 12
                if (country == "Cuba"):
                    x14 = 7150
                    # Iran, Honduras, Philippines, Italy, Poland,
                    # Jamaica, Vietnam,

                if (country == "Iran"):
                    x14 = 5430
                if (country == "Honduras"):
                    x14 = 15
                if (country == "Philippines"):
                    x14 = 3660
                if (country == "Italy"):
                    x14 = 31180
                if (country == "Poland"):
                    x14 = 12730
                if (country == "Jamaica"):
                    x14 = 19
                if (country == "Vietnam"):
                    x14 = 2160
                if (country == "Mexico"):
                    x14 = 8610
                    # Mexico, Portugal, Ireland, France,
                    # Dominican-Republic, Laos, Ecuador, Taiwan, Haiti,
                    # Columbia, Hungary, Guatemala, Nicaragua, Scotland,
                    # Thailand, Yugoslavia, El-Salvador,
                    # Trinadad&Tobago, Peru, Hong, Holand-Netherlands
                if (country == "Portugal"):
                    x14 = 19930
                if (country == "Ireland"):
                    x14 = 53370
                if (country == " France"):
                    x14 = 38160
                if (country == "Dominican-Republic"):
                    x14 = 52
                if (country == "Laos"):
                    x14 = 26
                if (country == "Ecuador"):
                    x14 = 5920
                if (country == "Taiwan"):
                    x14 = 28
                if (country == "Haiti"):
                    x14 = 29
                if (country == "Columbia"):
                    x14 = 30
                if (country == "Hungary"):
                    x14 = 12920
                if (country == "Guatemala"):
                    x14 = 32
                if (country == "Nicaragua"):
                    x14 = 33
                if (country == "Scotland"):
                    x14 = 34
                if (country == "Thailand"):
                    x14 = 5950
                if (country == "Yugoslavia"):
                    x14 = 36
                if (country == "El-Salvador"):
                    x14 = 37
                if (country == "Trinadad&Tobago"):
                    x14 = 38
                if (country == "Peru"):
                    x14 = 5960
                if (country == "Hong"):
                    x14 = 46310
                if (country == "Holand-Netherlands"):
                    x14 = 46910


                x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14])
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13)]
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)]
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9])
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8])
                #x = np.array([1, x1, x2, x3, x4, x5, x6, x7])
                #x = np.array([1, x1, x2, x3, x4, x5, x6])
                #x = np.array([1, x1, x2, x3, x4, x5])
                #x = np.array([1, x1, x2, x3, x4])
                #x = np.array([1, x1, x2, x3])
                #x = np.array([1, x1, x3])
                #self.X.append((x))
                #self.Y.append((s))
                #x = np.array([1, x1, x5, x13])
                if(errorNum < 32561):

                    # x = np.array([1, x1, x3, x5, x10, x11, x12, x13])
                    #print("hellooo")
                    self.X.append((x))
                    self.Y.append((s))
                    Xrow = x
                    Yrow = s
                    self.XY.append((Xrow, Yrow))
                if(errorNum >= 32561):
                    self.XYvalid.append((x, s))


    def svm(self):
        print("got here")
        #print(self.X)

        #self.X = np.reshape(self.X, (1, 15))
        #self.Y = np.reshape(self.Y, (1, 15))
        #clf = svm.SVC(gamma='scale')
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=.06, loss='deviance', subsample=1, min_samples_split=2, max_depth=2)          #.1304 error
        #clf = ExtraTreesClassifier()
        #clf = DecisionTreeRegressor(max_depth=2)
        #clf = RandomForestClassifier(n_estimators=100)# 13.96 error
        clf.fit(self.X, self.Y)

        error = 0
        for x,s in self.XYvalid:
            x = np.reshape(x, (-1, 15))
            prediction = 0
            if(clf.predict((x)) > 0):
                prediction = 1
            if (clf.predict((x)) < 0):
                prediction = -1
            #print(clf.predict((x)), "=", s)
            if(prediction != s):
                error = error + 1
        print("error after training: ", error/8140)
        #print("support vectors: ", clf.n_support_)
        #print("support vectors: ", clf.support_vectors_)
        #print("support vectors: ", clf.support_)


p = svmClass()
p.svm()
