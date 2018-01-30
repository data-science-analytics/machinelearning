import matplotlib.pyplot as plt
from numpy import *
from kNN import kNN


print('\ndatinglables……\n')
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')


print('\ndatamart20\n')
print(datingDataMat[0:20])


print('\n\n')
print(datingLabels[0:20])


print('\n\n')

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))

plt.show()


print('\n\n')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()


print('\n\n')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()


print('\n ranges \n')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print('\n\n')
print(normMat)
print('\n\n')
print(ranges)
print('\n\n')
print(minVals)


kNN.datingClassTest()


kNN.handwritingClassTest()
