import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import scipy.stats
import os
from scipy.io import loadmat, savemat
from math import atanh


# from torch.utils.data import DataLoader
# from data.read_data import MddData
#
# train_loader = DataLoader(dataset=MddData,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#


class MddData(DataLoader):
    # 初始化
    def __init__(self, MDD_dir, HC_dir, label_str='mdd'):
        super(MddData).__init__()
        print('label_str')
        print(label_str)
        self.label_str = label_str
        self.MDD_dir = MDD_dir  # MDD 文件路径
        self.HC_dir = HC_dir  # HC 路径
        self.MDD_path = os.listdir(MDD_dir)
        self.HC_path = os.listdir(HC_dir)
        self.MDD_path.sort(reverse=False)
        self.HC_path.sort(reverse=False)
        self.tc_data = np.zeros((1611, 116, 140), dtype=float)  # 获取时间序列，先清0
        self.age = np.zeros((1611), dtype=int)
        self.site = np.zeros((1611), dtype=int)
        self.sex = np.zeros((1611), dtype=int)
        self.label = np.zeros((1611), dtype=int)
        self.total_subjects = 1611  # 一共1611个被试
        self.time_length = 140  # 时间点设置成 140 (全部截取到140)

        for i in range(len(self.MDD_path)):  # 添加所有的 MDD 数据
            self.MDD_path[i] = os.path.join("/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/MDD",
                                            self.MDD_path[i])

        for i in range(len(self.HC_path)):  # 添加所有的 HC 数据
            self.HC_path[i] = os.path.join("/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/HC",
                                           self.HC_path[i])

        self.dataname_path = self.MDD_path + self.HC_path

        self.nrois = loadmat(self.MDD_path[0]).get('ROISignals').shape[1]  # 获取roi的个数

        "保存协变量的csv文件，站点，年龄，性别，诊断等"
        # covars = loadmat("../mydata/Covars.mat")
        # cov=covars['Covars']
        # columns = ["Site", "Age", "Sex", 'Diagnose']
        # dt = pd.DataFrame(cov, columns=columns)
        # dt.to_csv("../mydata/cov.csv", index=0)

        cov = pd.read_csv("/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/cov.csv")

        for i in range(len(self.dataname_path)):
            self.tc_data[i] = loadmat(self.dataname_path[i]).get('ROISignals').T[:, :self.time_length]  # 统一成ROI乘以时间点
            # self.data[i] = self.data[i][:,:140]   # 因为有的数据时间点最短为140，所以截取到140时间点
            self.age[i] = cov['Age'][i]
            self.site[i] = cov['Site'][i]
            self.sex[i] = cov['Sex'][i]
            self.label[i] = cov['Diagnose'][i]

        # 每个ROI上的时间序列 zscore 有的论文是以时间点为基准
        for i in range(len(self.tc_data)):
            self.tc_data[i] = scipy.stats.zscore(self.tc_data[i], axis=1)

        if self.label_str == 'mdd':
            # MDD是[0,1] HC是[1,0]
            self.label = np.eye(2)[self.label]

        elif self.label_str == 'sex':
            self.sex = np.eye(2)[self.sex]

    # 获取时间序列（zscore）
    def __getitem__(self, item):
        if self.label_str == 'mdd':
            return self.tc_data[item], self.label[item]
        elif self.label_str == 'sex':
            return self.tc_data[item], self.sex[item]

    # 获取时间序列的长度
    def __len__(self):
        return len(self.tc_data)

    def __getallitems__(self):
        if self.label_str == 'mdd':
            return self.tc_data, self.label
        elif self.label_str == 'sex':
            return self.tc_data, self.sex


class Mdd_corr(DataLoader):  # 将功能连接矩阵压平

    def __init__(self):
        super(Mdd_corr).__init__()

        # Initialize an np array to store all timecourses and labels
        # self.corr_data = np.zeros((data.total_subjects, N_corr_mat))
        self.graphs = []  # 要不要定义多个图，采用多种图来提取特征进行分类
        # self.fc_data = np.zeros((data.total_subjects,data.nrois,data.nrois))

        # Load data
        # print('Loading data & Creating graphs....')
        # for i in range(1528,data.total_subjects):
        #     fisherFc = np.zeros((116, 116), dtype=float)
        #     Fc = np.zeros((116,116),dtype=float)
        #
        #     for j in range(0, 116):
        #         for k in range(0, 116):
        #             X = data.tc_data[i][j,:]
        #             Y = data.tc_data[i][k,:]
        #
        #             cc = np.array([X, Y])
        #             cc_pd = pd.DataFrame(cc.T, columns=['c1', 'c2'])
        #             cc_corr = cc_pd.corr(method='pearson')  # 相关系数矩阵
        #             val = cc_corr.iloc[0, 1]
        #             Fc[j][k] = val
        #             if(j == k):
        #                 fisherFc[j][k] = -1
        #             else:
        #                 fisherFc[j][k] = atanh(val)
        #
        #     corr_vals = fisherFc.copy()
        #     cc_triu_ids = np.triu_indices_from(corr_vals,k=1)  # 取上三角矩阵  对角线不要
        #     cc_vector = corr_vals[cc_triu_ids]
        #     self.corr_data[i] = cc_vector
        #     self.fc_data[i] = Fc.copy()
        # savemat("../mydata/fisherFc/fisherFc{:0>4d}.mat".format(i),{'fisherFc':cc_vector})
        # savemat("../mydata/Fc/Fc{:0>4d}.mat".format(i),{'Fc':Fc})

        # for i in range(1611):
        #     self.corr_data[i] = loadmat("../mydata/fisherFc/fisherFc{:0>4d}.mat".format(i)).get('fisherFc')
        #     self.fc_data[i] = loadmat("../mydata/Fc/Fc{:0>4d}.mat".format(i)).get('Fc')

        # savemat("../mydata/fisherFc.mat",{'fisherFc':self.corr_data}) # 计算太耗时了，所以保存下来,保存压平的fisher功能连接特征
        # savemat("../mydata/Fc.mat",{'Fc':self.fc_data})

        # self.corr_data = loadmat("../mydata/fisherFc.mat").get('fisherFc')

        m = loadmat("../../mydata/removeCovCombatfisherFC.mat")
        data = m.get('removeCovCombatfisher')
        data = data.T
        m2 = loadmat("../../mydata/removeCovCombatNetwork.mat")
        data2 = m2.get('removeCovCombatNetwork')
        data2 = data2.T

        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer()
        data2 = scaler.fit_transform(data2)
        data = np.concatenate((data, data2), axis=1)  # 1611 * 7842

        # importantSvmFeat = loadmat(
        #     '../mydata/importantSvmFeat.mat').get('Feat')
        # importantSvmFeat = importantSvmFeat[0]  # 这里面保存的是重要的特征索引号
        #
        # self.corr_data = []
        # for i in importantSvmFeat:  # 重要特征索引号
        #     self.corr_data.append(data[:, i])
        # self.corr_data = np.asarray(self.corr_data)
        # self.corr_data = self.corr_data.T

        self.corr_data = data

        data = MddData("../../mydata/MDD", "../mydata/HC")
        self.nrois = data.nrois
        # N_corr_mat = int(data.nrois * (data.nrois - 1) / 2)

        self.label = data.label  # 生成one-hot标签  比如第一个人是病人，所以是[0,1]  后面的是正常人所以是[1,0]

    def __len__(self):
        return len(self.corr_data)

    def __getitem__(self, index):
        return self.corr_data[index], self.label[index]

    def __getallitems__(self):
        return self.corr_data, self.label


class selected_MddData(DataLoader):
    def __init__(self, MDD_dir, HC_dir, num_roi=116, time_length=140):
        super(selected_MddData).__init__()
        self.mdd_filelist = os.listdir(MDD_dir)
        self.hc_filelist = os.listdir(HC_dir)
        self.mdd_filelist.sort(reverse=False)
        self.hc_filelist.sort(reverse=False)
        self.total_sub = len(self.mdd_filelist) + len(self.hc_filelist)
        self.time_length = time_length
        self.num_roi = num_roi

        self.tc_data = np.zeros((self.total_sub, num_roi, time_length), dtype=float)
        self.labels = np.zeros((self.total_sub), dtype=int)

        for i in range(len(self.mdd_filelist)):
            self.mdd_filelist[i] = os.path.join(MDD_dir, self.mdd_filelist[i])

        for i in range(len(self.hc_filelist)):
            self.hc_filelist[i] = os.path.join(HC_dir, self.hc_filelist[i])

        i = 0
        for j in range(len(self.mdd_filelist)):
            self.tc_data[i] = loadmat(self.mdd_filelist[j]).get('ROISignals').T[:, :self.time_length]
            self.tc_data[i] = scipy.stats.zscore(self.tc_data[i], axis=1)
            self.labels[i] = 1
            i = i + 1

        for j in range(len(self.hc_filelist)):
            self.tc_data[i] = loadmat(self.hc_filelist[j]).get('ROISignals').T[:, :self.time_length]
            self.tc_data[i] = scipy.stats.zscore(self.tc_data[i], axis=1)
            self.labels[i] = 0
            i = i + 1

        self.labels = np.eye(2)[self.labels]

    def __len__(self):
        return self.total_sub

    def __getitem__(self, item):
        return self.tc_data[item], self.labels[item]

    def __getallitems__(self):
        return self.tc_data, self.labels





if __name__ == "__main__":
    """
    前832个人是抑郁症病人，后面779个人是正常人,测试无问题，时间序列数据经过了zscore
    """
    #     a = MddData("../mydata/MDD", "../mydata/HC")  # 获取每个人zscore之后的时间序列数据和一些协变量数据，标签没有问题
    #     a.__getitem__(0)
    #     data, label = a.__getitem__(0)

    #     b = Mdd_corr()  # 获取每个人压平的功能连接矩阵和标签
    #     data, label = b.__getitem__(0)

    #     c = Mdd_brainetcnn()
    #     data, label = c.__getitem__(0)

    # adhdData = ADHDData()
    # print(adhdData.total_subject)

