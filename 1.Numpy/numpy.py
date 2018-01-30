# coding=gbk

from numpy import *


print(random.rand(4, 4))


randMat = mat(random.rand(4, 4))
print(randMat)


invRandMat = randMat.I
print(invRandMat)


multi = invRandMat*randMat
print(multi)


contrast = multi - eye(4)
print(contrast)
