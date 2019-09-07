#泰坦尼克号生还者预测
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
train_data = pd.read_csv('F:/taitannike/titanic/train.csv')
test_data = pd.read_csv('F:/taitannike/titanic/test.csv')

train_data.info()   #查看相关数据信息（每列数据总数，每列数据有无空值之类）
test_data.info()
#此处运行一次，发现train_test一共有891行数据，但是age只有714行数据，Cabin更少，只有204行数据，Embarked有889行数据，也少两行
#test_data一共有418行数据，但是age只有332行数据，Fare只有417行数据，Cabin只有91行数据，
#对这些数据缺失的特征值而言，如何处理他们是一个大问题，关系着模型的准确度train_data.describe()

#查看最大值，最小值，百分数之类的数据，与如何处理缺失值有很大关系（name,sex,ticket,cabin,embarked这些不是数字的在处理过程中都被忽略了
#可以看出，至少百分之五十以上的人Pclass都是3，平均年龄接近30岁，至少50%以上的人旅途中没有同乘的兄弟姐妹或者配偶，70%以上的人旅途中没有同乘的
#父母或者小孩。另外，对比平均值mean与中位数50%对应的值，可以观察出数据的大概分布。比如说，age的mean和50%很接近，说明左右分布对称，fare的mean
#和50%相差很大，说明位于中位值之后的数据比位于中位数前的数据少得多，大多数人都坐的是比较便宜的位置
#平均值mean的计算

train_data.describe()

#查看最大值，最小值，百分数之类的数据，与如何处理缺失值有很大关系（name,sex,ticket,cabin,embarked这些不是数字的在处理过程中都被忽略了
#可以看出，至少百分之五十以上的人Pclass都是3，平均年龄接近30岁，至少50%以上的人旅途中没有同乘的兄弟姐妹或者配偶，70%以上的人旅途中没有同乘的
#父母或者小孩。另外，对比平均值mean与中位数50%对应的值，可以观察出数据的大概分布。比如说，age的mean和50%很接近，说明左右分布对称，fare的mean
#和50%相差很大，说明位于中位值之后的数据比位于中位数前的数据少得多，大多数人都坐的是比较便宜的位置
#平均值mean的计算

test_data.describe(include='all')   #include='all'包括了所有的信息，有可能对特征值的选择有很大用处，比如top（频数最高者）。还有看unique这一行
#我们可以知道，name,ticket,cabin这几列值都很多，因此很难作为特征值（初步这样认为），其余的sex,embarked都是有数的，好分类，因此可以使用

train_data.head()   #查看数据具体信息，其中passengerId只是一个编号，应该与survived无关，Name不仅是字符而且毫无规律，因此初步认为无关，Sex列有
#两类，分成male与female，因此需要进行one-hot编码，后续的Embarked也是如此

#现在可视化分析特征与结果的关系
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.plot(train_data['PassengerId'],train_data['Survived'].cumsum())
ax1.set_title('PassengerId--Survived')
ax1.set_xlabel('PassengerId')
ax1.set_ylabel('all_number_of_survived')

#在图中，我们可以清楚的看到，乘客的id和总的获救的人数基本是成一条直线，因此可以认为两者无关，passengerId不作为特征值。
#其实我的理解就是这个Id编号应该是我们后来者自己家的，因此肯定是与结果无关的

#接下来分析Pclass，Sex,Age,SibSp,Parch,Fare,Embarked这几个特征与结果Survived的关系，Cabin这个特征最后分析
fig = plt.figure(figsize=(25,20))
ax1 = fig.add_subplot(2,4,1)
#这种画图最好选择seaborn画，相比于matplotlib功能更强大，也更简单
#对Pclass画图，因为Pclass值很少，因此可用卡方图画出它与Survived的关系
#可以看出，Pclass越小（等级越高），获救的概率越大(1与0 的比值），在pclass=3的时候，概率更是大幅度下降
sns.countplot(x='Pclass',hue='Survived',data=train_data)

#对Sex画图，和Pclass一样，使用卡方图
#从图看出，女性获救的概率远大于男性
ax2=fig.add_subplot(2,4,2)
sns.countplot(x='Sex',hue='Survived',data=train_data)

#对SibSp画图，Age值很多，不适用卡方图，稍后使用密度图画
#从图看出，在sibsp=1或者2的时候获救的概率更大，它的值对于结果是有影响的
ax3 = fig.add_subplot(2,4,3)
sns.countplot(x='SibSp',hue='Survived',data=train_data)

#对Parch画图
#从图看出，在Parch=1，2的时候获救的概率更大，它的值对于结果是有影响的
ax4 = fig.add_subplot(2,4,4)
sns.countplot(x='Parch',hue='Survived',data=train_data)

#对Embarked画图
#从图可以看出，在C港口上船的人获救概率最大，S港口上船的人获救概率最小
ax5 = fig.add_subplot(2,4,5)
sns.countplot(x='Embarked',hue='Survived',data=train_data)

#对age画图.这里我们不仅需要找到年龄与结果Survived的关系，同样，最好还有获救的人和没获救的人的年龄对比。如下
#从结果上看，明显整体而言，中间年龄段的人获救和没获救的概率都很大，这应该主要是因为这个年龄段的人比较多，因此，单纯的分析一条线没有意义
#对比获救的人和没获救的人，我们可以看出获救的（红线）0-10这个年龄段的人相比没获救的人概率要大，因此，年龄和结果相关
ax6 = fig.add_subplot(2,4,6)
sns.kdeplot(train_data.loc[train_data['Survived'] == 0,'Age'],color='k',shade=True,label='not survived')
sns.kdeplot(train_data.loc[train_data['Survived'] == 1,'Age'],color='r',shade=True,label='survived')
ax6.set_xlabel('age')
ax6.set_ylabel('frequency')

#同理，对Fare画图.同样，可以看出票价在30-60附近的最多，价格越高，红线（获救的人）的概率越大，价格在降低的时候，黑线（未获救）的人的概率远超红线
ax7 = fig.add_subplot(2,4,7)
sns.kdeplot(train_data.loc[train_data['Survived'] == 0,'Fare'],color='k',shade=True,label='not survived')
sns.kdeplot(train_data.loc[train_data['Survived'] == 1,'Fare'],color='r',shade=True,label='survived')
ax7.set_xlabel('fare')
ax7.set_ylabel('frequency')

#分析Cabin这个特征，这个特征值很杂，而且还有大量缺失，因此简单的方法是直接将拥有cabin的记为yes，没有的记为no，实际上还可能与每个客舱里面的作为
#有关，这个可以留待以后来考虑。
#可以看出，have_cabin中，获救的人数是未获救的人数的两倍左右，而在no_cabin中，恰恰相反，没获救的人的数量是获救的人的数量的一倍多，因此cabin对
#Survived的结果有很大影响，把它作为特征值考虑。
ax8 = fig.add_subplot(2,4,8)
have_cabin = train_data.Survived[pd.notnull(train_data.Cabin)].value_counts()
print(have_cabin) #查看have_cabin的结果
no_cabin = train_data.Survived[pd.isnull(train_data.Cabin)].value_counts()
print(no_cabin)   #查看no_cabin的结果)
df = pd.DataFrame({'have':have_cabin,'no':no_cabin})
df.plot(kind='bar',ax=ax8)
df

#通过以上7个图，我们可以发现，这7个特征与Surviver的结果都是有关系的，因此全部作为特征考虑

#现在对缺失值进行处理。首先数据缺失的一共有3种，分别是age,cabin,embarked
#首先Embarked的数据缺失的最少，首先处理这列，直接用出现频率最高的值填充缺失值就好。但不能简单的用全体的出现频率，要考虑特征
train_data[train_data['Embarked'].isnull()]
#可以观察得到，两人survived，pclass都为1，sex都为female，sibsp，survived都为0，fare，cabin，甚至ticket都相同，因此可以认为两人的embarked都相同
#下面寻找具有相同特征的样本,注意特征不能全选，选择一些最有代表性的就好
test1=train_data.loc[(train_data['Pclass']==1)&(train_data['Sex']=='female')&(train_data['SibSp']==0)&(train_data['Parch']==0)]  #注意每个条件之间的小括号不能省略

test1.describe(include='all')  #我们可以看到满足条件的一共有32种，出现频率最高的是C，所以给Embarked填充C
train_data['Embarked']=train_data['Embarked'].fillna('C')

#接下来填充age列的缺失值，同样的，因为age列缺失数值较多，所以考虑填充考虑了特征之后的平均值
age_group_mean = train_data.groupby(['Pclass','Sex','Embarked'])['Age'].mean().reset_index()
def age_fill(row):
    return age_group_mean[((row['Pclass']==age_group_mean['Pclass'])&(row['Sex']==age_group_mean['Sex'])
                           &(row['Embarked']==age_group_mean['Embarked']))]['Age']  #注意此时age_group_mean中接的是一个Series格式的布尔数组，按行取值
#接下来就是填充age缺失值了

train_data['Age'] = train_data.apply(lambda x:age_fill(x) if np.isnan(x['Age']) else x['Age'],axis=1)  #此时x是样本数组中的一行，x['Age']是一个数字
train_data.info()   #看一下结果
#这一步我们把age列的缺失值填好了

#现在考虑填充最后一个缺失值Cabin列。因为Cabin缺失值很多，因此不能仔细考虑，在此简单的将他们分为有（have）和无（no）。
#同样的原因，需要考虑test_data，因此在此使用函数
def set_cabin_type(dataframe):
    dataframe.loc[(dataframe.Cabin.isnull()),'Cabin'] = 'no'
    dataframe.loc[(dataframe.Cabin.notnull()),'Cabin'] = 'yes'
    return dataframe
dataframe=train_data
train_data=set_cabin_type(dataframe)
train_data.info()  #看看现在整体情况

train_data.head()  #看看现在数据的具体情况，我们发现age，fare这些数据都很大，pclass这些数据却很小，这会导致两者之间权重不同，因此，需要对
#他们进行标准化处理。sex，cabin，embarked这几列都是字符串而不是数字，因此也需要对他们进行热编码

#对pclass，sex，sibsp，parch，cabin，embarked这些属性进行热编码
dummies_Pclass = pd.get_dummies(train_data['Pclass'],prefix='Pclass')
#dummies_SibSp = pd.get_dummies(train_data['SibSp'],prefix='SibSp')  特征量太多，先不用
#dummies_Parch = pd.get_dummies(train_data['Parch'],prefix='Parch')  特征量太多，先不用
dummies_three = pd.get_dummies(train_data[['Sex','Cabin','Embarked']],prefix=['Sex','Cabin','Embarked'])
train_data_result = pd.concat([train_data,dummies_Pclass,dummies_three],axis=1)
train_data_result.drop(['Pclass','Sex','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)
#axis=1不要忘了，除了对dataframe进行索引是直接在列上以外，其余的都默认在行上
#看一下具体的数据
train_data_result.head()

#现在标准化数据，引入StandardScaler模块
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_data_result[['Age','SibSp','Parch','Fare']])     #此处必须而且只能选择age，fare这两列数据进行标准化，这个数据test_data标准化的时候也会用到
train_data_result[['Age','SibSp','Parch','Fare']] = scaler.transform(train_data_result[['Age','SibSp','Parch','Fare']])
train_data_result.head()  #看看结果

#用逻辑回归建模
from sklearn.linear_model import LogisticRegression   #引入逻辑回归模块
last_train_data = train_data_result.drop(['PassengerId','Survived'],axis=1,inplace=False)  #除去PassengerId，Survived这两个特征，这个作为训练数据
goal_train_data = train_data_result['Survived'] #作为训练数据的目标
model = LogisticRegression()
model.fit(last_train_data,goal_train_data)
model #看看自己创建的模型

#接下来对test_data作和train_data一样的数据处理
#同样，最初观察可知，test_data有三个特征数据缺失，分别是Fare,Cabin,Age。Fare只缺少一个，age缺少86个，Cabin缺少327个
test_data[test_data['Fare'].isnull()]  #已知重要的特征为pclass，sex，age，sibsp，parch，embarked,但是显然age不能作为确定特征，但可以以一个范围作为特征

test2=test_data.loc[(test_data['Pclass']==3)&(test_data['Sex']=='male')&(test_data['Age']>40)&(test_data['SibSp']==0)
                    &(test_data['Parch']==0)&(test_data['Embarked']=='S')]  #注意每个条件之间的小括号不能省略

test2.describe(include='all')  #我们可以看到满足条件的一共有4种，fare的均值是10.28，我们就给fare的缺失值为10.28
test_data['Fare']=test_data['Fare'].fillna(10.28)
test_data.head()

#这一步处理age列数据，因为age列缺失数值较多，所以考虑填充考虑了特征之后的平均值
age_group_mean2 = test_data.groupby(['Pclass','Sex','Embarked'])['Age'].mean().reset_index()

#直接使用train_test数据处理时候建好的age_fill函数。然后填充age缺失值

test_data['Age'] = test_data.apply(lambda x:age_fill(x) if np.isnan(x['Age']) else x['Age'],axis=1)  #此时x是样本数组中的一行，x['Age']是一个数字
test_data.info()   #看一下结果
#这一步我们把age列的缺失值填好了

#这一步处理cabin数据,使用既有的set_cabin_type函数
dataframe2=test_data
test_data=set_cabin_type(dataframe2)
test_data.info()  #看看现在整体情况
test_data.head()   #看看具体情况

#对test_data数据进行独热编码
#对pclass，sex，sibsp，parch，cabin，embarked这些属性进行热编码
dummies2_Pclass = pd.get_dummies(test_data['Pclass'],prefix='Pclass')
dummies2_three = pd.get_dummies(test_data[['Sex','Cabin','Embarked']],prefix=['Sex','Cabin','Embarked'])
test_data_result = pd.concat([test_data,dummies2_Pclass,dummies2_three],axis=1)
test_data_result.drop(['Pclass','Sex','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)
#axis=1不要忘了，除了对dataframe进行索引是直接在列上以外，其余的都默认在行上
#看一下具体的数据
test_data_result.head()

#现在标准化test_data数据，必须使用早已计算好的标准化模块

test_data_result[['Age','SibSp','Parch','Fare']] = scaler.transform(test_data_result[['Age','SibSp','Parch','Fare']])
test_data_result.head()  #看看结果

#最后一步，数据预测
last_test_data = test_data_result.drop('PassengerId',axis=1,inplace=False)
predictions = model.predict(last_test_data)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values,'Survived':predictions.astype(np.int)})
result.to_csv("F:/taitannike/titanic/predictions_result.csv",index=False)