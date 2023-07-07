import os
from copy import deepcopy

import shap
import torch
from matplotlib import pyplot as plt
from torch import nn

from visualization_model import TSTransformerEncoderClassiregressor
import numpy as np
from torch.autograd import Variable

from scipy.stats import ttest_ind



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
data_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz'
dataset = np.load(data_path)
datas = dataset['tc']  # (1611, 116, 116)
labels = dataset['labels']

'''
datas = np.random.random((1611, 116, 116))
labels = np.ones((1611, 2))
# '''

zeros_data = np.zeros((1611, 128, 116))
zeros_data[:, :116, :] = datas
datas = zeros_data


# 初始化字符串

# 创建一个 116x116 的二维数组，其中每个元素都包含指定的字符串
array_116x116 = [[f"[{i + 1},{j + 1}]" for j in range(116)] for i in range(116)]
feature_names = np.array(array_116x116)

rps = 1
folds = 1

path = './5rp_5fold_shap_values.npy'
# path = './5rp_5fold_shap_values(30epochs_9950featrues).npy'
# 加载训练好的PyTorch模型
if not os.path.exists(path):
    max_list = []
    all_shap_values = {}
    for rp in range(rps):
        for fold in range(folds):
            # model = AutoencoderClassifier(num_featrues, num_featrues // 2, 1)
            # model.load_state_dict(torch.load(f'./60epochs_no_FS/best_model_on_everyfold/rest-meta-mdd_ResDAE_CC200/repeat{rp}/model_fold{fold}.pt'))

            model = TSTransformerEncoderClassiregressor(feat_dim=116, max_len=116, d_model=128,
                                                        n_heads=8,
                                                        num_layers=3, dim_feedforward=256,
                                                        num_classes=2,
                                                        dropout=0.1, pos_encoding='fixed',
                                                        activation='gelu',
                                                        norm='BatchNorm',
                                                        num_linear_layer=1)
            # 加载训练好的PyTorch模型
            '''
            start_epoch = 0
            model_path = f'/home/studio-lab-user/sagemaker-studiolab-notebooks/transformer_code/experiments/MDD_allsub_fc_train_2023-02-24_10-03-04_3l5/checkpoints/model_best.pth'
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict = deepcopy(checkpoint['state_dict'])
            # if change_output:
            #     for key, val in checkpoint['state_dict'].items():
            #         if key.startswith('output_layer'):
            #             state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)  # if args.use_cuda:
            #     model = model.cuda().eval()
            '''

            model.to(device)
            model.eval()

            # 初始化SHAP解释器
            print("初始化SHAP解释器中......")
            shap.initjs()

            example_data = torch.randn(32, 128, 116)
            explainer = shap.DeepExplainer(model, Variable(example_data).to(device))

            # SHAP值
            print("计算SHAP值中......")
            for data in datas:
                data = torch.from_numpy(data.astype('float32')).unsqueeze(0)
                # background_samples = shap.sample(val_data, 10)
                # shap_values:(124, 9950)
                shap_values = explainer.shap_values(Variable(data).to(device))
                shap_abs_mean = np.array([np.mean(np.abs(shap_values[0][:, i])) for i in range(shap_values[0].shape[1])])

                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
                feature_order = feature_order[-min(20, len(feature_order)):]
                max_list.append(feature_names[feature_order])

                all_shap_values[f'rp{rp}fold{fold}'] = shap_values

    arrays = np.array(max_list)
    np.savez('./max_20_list.npz', arrays)

    # np.save('./5rp_5fold_shap_values.npy', all_shap_values)

    np.save(path, all_shap_values)
else:
    all_shap_values = np.load(path, allow_pickle=True).item()

print(all_shap_values['rp0fold0'].shape)

arrays = np.load('./max_20_list.npz')['arr_0']

k = 2  # 至少出现在 k 个数组中

# 使用字典统计元素的出现次数
counts = {}
for arr in arrays:
    for x in set(arr):
        counts[x] = counts.get(x, 0) + 1

# 打印出现次数至少为 k 的元素
result = [x for x, count in counts.items() if count >= k]
if result:
    print(f"在至少 {k} 个数组中出现过的元素有：")
    print(result)
else:
    print(f"没有元素在至少 {k} 个数组中出现过。")

for rp in range(rps):
    for fold in range(folds):
        from sklearn.feature_selection import SelectKBest, f_classif

        # # datas是特征矩阵，labels是标签
        # selector = SelectKBest(score_func=f_classif, k=num_featrues // 2)  # 在此处选择 fisher score
        # X_new = selector.fit_transform(datas, labels)
        # regs = selector.get_support(indices=True)
        # datas = np.array([datas[i][regs] for i in range(datas.shape[0])])

        shap_values = all_shap_values[f'rp{rp}fold{fold}']

        # 可视化SHAP值
        print("可视化SHAP值中......")
        plt.figure(figsize=(10, 10), dpi=500)
        # plt.title(f'fold {fold} dot')
        shap.summary_plot(shap_values, datas.astype('float32'), feature_names,
                          plot_type="dot", show=False,
                          max_display=20, plot_size=(10, 10))
        plt.tight_layout()
        plt.savefig(f"./Data/shapPictrue/19900features/repeat{rp}/repeat{rp}fold{fold}Figure A.{0}.(a).tif", dpi=500,
                    format='tif')
        plt.show()

        plt.figure(figsize=(10, 10), dpi=500)
        # plt.title(f'fold {fold} bar')
        shap.summary_plot(shap_values, datas.astype('float32'), feature_names,
                          plot_type="bar", show=False,
                          max_display=20, plot_size=(10, 10))
        plt.savefig(f"./Data/shapPictrue/19900features/repeat{rp}/repeat{rp}fold{fold}Figure A.{0}.(b).tif", dpi=500,
                    format='tif')
        plt.tight_layout()
        plt.show()
        plt.close('all')

# kernelexplainer

# f = lambda x: model(Variable(torch.from_numpy(x.astype('float32')))).detach().numpy()
# explainer = shap.KernelExplainer(f, datas.astype(float))
# shap_values = explainer.shap_values(val_data.astype(float))
# shap.initjs()
#
# shap.summary_plot(shap_values[0], val_data.astype(float), feature_names[regs])
