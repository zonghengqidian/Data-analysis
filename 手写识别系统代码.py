'''
使用k近邻算法实现手写识别系统
'''
import numpy as np
from os import listdir
'''
k近邻算法

Input:      inX: 一个测试数据，是(1xN)的数组
            dataSet: 样本数据的特征矩阵，是(NxM)的数组，其中N为样本数据的个数，M是每个样本具有的特征数
            labels: 已知数据的标签或类别，是(1xM）的数组
            k: k近邻算法中的k，含义是选取距离最近的k个点          
Output:     测试样本所属的标签，属于labels中的一个

'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # shape[0]返回dataSet的行数，也就是样本数据个数N
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    # np.tile(inX,(a,b))函数将inX重复a行，重复b列
    sqDiffMat = diffMat**2
    #作差后,对数组中每个值求平方
    sqDistances = sqDiffMat.sum(axis=1)
    #np.sum()是对数组中所有元素求和，sum(axis=0)是每列所有元素求和，sum(axis=1)是每行所有元素求和
    distances = sqDistances**0.5
    #开平方，求欧式距离
    sortedDistIndicies = distances.argsort()
    #np.argsort()函数返回的是数组中值从小到大排序所对应的在还未排序的数组中的索引值，这个很重要，最好在自己电脑上打开ipython自己试一下
    classCount={}
    #建立一个空字典，存储对这个测试数据的判断结果
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #取出前k个距离测试点最近的点对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #计算每个标签的样本数。字典get()函数返回指定键的值，如果值不在字典中返回默认值0，与dict.key()不一样
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    #reverse=True为降序排列，不写的话都是默认升序排列字，这时结果是一个由键值对组成的元组组成的列表，也就是列表中包含元组，元组中包含有键值对
    return sortedClassCount[0][0] #返回列表的第一个元组的第一个值，也就是测试样本所属的标签

'''
函数功能：将（32x32）的二进制图像转换为长1024的一维数组

Input:     filename :文件名
Output:    长1024的一维数组

'''
def img2vector(filename):                       #第一种方法
    fr = open(filename)
    first_str = fr.read()
    second_str = first_str.split()
    last_str = ''.join(second_str)
    returnVect1 = list(last_str)
    returnVect = [int(i) for i in returnVect1]  #这句话必须要有，不然列表中的数字会有引号，导致出现dtype<'U1'的情况，最终程序将无法运行
    return returnVect
#def img2vector(filename):                      #第二种方法
#    returnVect = np.zeros((1,1024))            #创建空numpy数组，是一个（1×10）的二维数组
#    fr = open(filename)                         #打开文件，只要不使用close语句，文件就一直会打开
#    for i in range(32):
#        lineStr = fr.readline()                #读取每一行内容，返回的是str，readlines()返回的是字典，不可搞混了
#        for j in range(32):
#            returnVect[0,32*i+j] = int(lineStr[j])#将每行前32个字符值存储在numpy数组中
#    return returnVect

'''
函数功能：手写数字分类测试
'''
def handwritingClassTest(filename,filename2):
    hwLabels = []                           #建立一个存储标签（labels）的列表，便于索引
    trainingFileList = listdir(filename)    #加载训练集,返回的是个列表，得到的是filename文件夹下面文件的列表
    m = len(trainingFileList)                     #计算文件夹下文件的个数，因为每一个文件是一个手写体数字，每一个文件都对应着特征与标签
    trainingMat = np.zeros((m,1024))            #初始化训练数组（样本数组），大小为（M,N）的数组
    for i in range(m):
        fileNameStr = trainingFileList[i]        #获取文件名
        fileStr = fileNameStr.split('.')[0]     #从文件名中解析出分类的第一个结果，
        classNumStr = int(fileStr.split('_')[0]) #根据实际情况，从文件名通过split分类，又通过'_'分类，最后得到这个测试数据到底是0-9中哪个标签
        hwLabels.append(classNumStr)             #将得到的标签添加到最开始存储标签的列表hwLabels中
        trainingMat[i,:] = img2vector(filename+'/%s' % fileNameStr)  #对每一个样本数组，添加相应的特征值
    testFileList = listdir(filename2)          #加载测试数据集，结果是个列表。下面过程是一个验证算法准确度的过程
    errorCount = 0.0                           #常用的计数方法
    mTest = len(testFileList)                  #获取测试数据组合的长度，返回int
    for i in range(mTest):
        fileNameStr = testFileList[i]             #第i个位置处的文件名称
        fileStr = fileNameStr.split('.')[0]          #将文件名按照split('.'')分裂后形成一个新的列表，再选前者
        classNumStr = int(fileStr.split('_')[0])     #将上步得到的结果按照'_'进行分割，得到测试数据对应的真是标签，便于两者之间比较
        vectorUnderTest = img2vector(filename2+'/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #开始分类
        print ('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0            #计算分错的样本数
    print ('\nthe total number of errors is: %d' % errorCount)
    print ('\nthe total error rate is: %f' % (errorCount/float(mTest)))  #错的样本数量与实际进行测试的数据数量的比值，就是错误的概率，也是算法的正确度

'''
主函数
'''
if __name__ == '__main__':
    filename = 'F:/BaiduNetdiskDownload/机器学习实战源代码MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits'#文件的绝对路径
    filename2 = 'F:/BaiduNetdiskDownload/机器学习实战源代码MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits'  #文件的绝对路径
    handwritingClassTest(filename,filename2)


#最后的结果
#the total number of errors is: 10
#the total error rate is: 0.010571
















