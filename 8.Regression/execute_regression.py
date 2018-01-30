from numpy import *
import matplotlib.pyplot as plt
from Regression import regression

xArr, yArr = regression.loadDataSet('ex0.txt')

print(xArr[0:10])
print(yArr[0:10])

ws = regression.standRegres(xArr, yArr)
print(ws)

xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()


yHat = xMat * ws
print(corrcoef(yHat.T, yMat))


xArr, yArr = regression.loadDataSet('ex0.txt')
print(xArr[0])
regression.lwlr(xArr[0], xArr, yArr, 1.0)
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.02)

strInd = xMat[:, 1].argsort(0)
xSort = xMat[strInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[strInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()


abX, abY = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(abX, abY)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()


xArr, yArr = regression.loadDataSet('abalone.txt')
final = regression.stageWise(xArr, yArr, 0.01, 300)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(final)
plt.show()
