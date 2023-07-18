import argparse
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
# import timm
from visualization_model import TSTransformerEncoderClassiregressor as mymodel
from collections import OrderedDict

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    # batch_size = lengths.numel()
    # max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    # return (torch.arange(0, max_len, device=lengths.device)
    #         .type_as(lengths)
    #         .repeat(batch_size, 1)
    #         .lt(lengths.unsqueeze(1)))
    return torch.zeros(lengths) == torch.zeros(lengths)


# 定义一些参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--model', type=str, default='recurrent', choices={'all', 'recurrent'},
                        help='select the model to visualization.')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='the threshold about the cam')
    parser.add_argument('--method', type=str, default='gradcam++',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=16, width=16):  # 116，1，128
    # print('input tenor shape:{}'.format(tensor.shape))
    tensor = tensor.permute(1, 0, 2)  # 1，116，128

    result = tensor.reshape(tensor.size(0),
                            height, width, -1)  #

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image> -image_name <the_name_of_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")


    # todo: define the model of yourself
    # todo: set the model setting file_path

    '''test the random model'''
    model = mymodel(feat_dim=116, max_len=116, d_model=128,
                    n_heads=8,
                    num_layers=3, dim_feedforward=256,
                    num_classes=2,
                    dropout=0.1, pos_encoding='fixed',
                    activation='gelu',
                    norm='BatchNorm',
                    num_linear_layer=1)
    loaded_models = []
    path_start = r'/home/studio-lab-user/'
    path_end = r'/checkpoints/model_best.pth'
    if args.model == 'all':
        model_paths = []  # length: 10, all the trained models path
    if args.model == 'recurrent':
        model_paths = []  # length: 10, all the trained models path

    for model_path in model_paths:
        # '''
        start_epoch = 0
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = deepcopy(checkpoint['state_dict'])
        # if change_output:
        #     for key, val in checkpoint['state_dict'].items():
        #         if key.startswith('output_layer'):
        #             state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)  # if args.use_cuda:
        #     model = model.cuda().eval()
        if args.use_cuda:
            model = model.cuda().eval()
            # '''
        print(model)
        loaded_models.append(model)

    # # '''
    # start_epoch = 0
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # state_dict = deepcopy(checkpoint['state_dict'])
    # # if change_output:
    # #     for key, val in checkpoint['state_dict'].items():
    # #         if key.startswith('output_layer'):
    # #             state_dict.pop(key)
    # model.load_state_dict(state_dict, strict=False)  # if args.use_cuda:
    # #     model = model.cuda().eval()
    # if args.use_cuda:
    #     model = model.cuda().eval()
    #     # '''

    # print(dir(model))
    # target_layer = model.block4[-1].norm1

    # todo: set the target model layer
    cams = []

    for model in loaded_models:
        target_layer = model.transformer_encoder.layers
        # print(target_layer)

        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        cam = methods[args.method](model=model,
                                   target_layers=target_layer,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)
        cams.append(cam)


    # deal with the input picture
    # todo: deal with the input data, to change it to a tensor

    '''random test'''
    max_len = 128
    # input = np.random.random((116, 116))
    # input_tensor = torch.zeros(1, max_len, input[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    # input_tensor[0, :116, :] = torch.tensor(input)
    # input_numpy = np.float32(torch.tensor(input).unsqueeze(2).expand(116, 116, 3)) / 255

    # '''

    all_grayscale_cam_list = []
    print('subjects is all')
    data_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz'
    dataset = np.load(data_path)
    datas = dataset['tc']  # (1611, 116, 116)
    labels = dataset['labels']
    print('{} subjects will be used for visualization'.format(datas.shape[0]))
    for cam in cams:
        for data in datas:
            all_grayscale_cam = np.zeros((datas.shape[1], datas.shape[2]), dtype=float)
            input = torch.tensor(data)
            # print('input data:{}'.format(input.shape))
            # print(input)
            input_tensor = torch.zeros(1, max_len, data.shape[-1])  # (batch_size, padded_length, feat_dim)
            input_tensor[0, :116, :] = input
            # print('input tensor:')
            # print(input_tensor)
            # input_numpy = np.ones((128, 116, 3), dtype=float)
            input_numpy = np.float32(input.unsqueeze(2).expand(116, 116, 3)) / 255
            # '''

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = 2
            # print(target_category)
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1

            grayscale_cam = cam(input_tensor=input_tensor,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :116, :]
            # print(grayscale_cam)
            all_grayscale_cam = all_grayscale_cam + grayscale_cam
        print('all subjects is all ready ok.')
        all_grayscale_cam = all_grayscale_cam / datas.shape[0]
        all_grayscale_cam_list.append(all_grayscale_cam)

    results_all = np.array(all_grayscale_cam_list)

    np.savez(f'{args.method}_{args.model}_results.npz',
             results=results_all)


    ''' save results and image '''
    # cam_image = show_cam_on_image(input_numpy, all_grayscale_cam)
    # cv2.imwrite(f'PVT_{args.method}_' + args.subjects + '_no_threshold.jpg', cam_image)
    # all_grayscale_cam_no_threshold = all_grayscale_cam
    #
    # np.savez(f'{args.method}_' + args.subjects + '_no_threshold.npz',
    #          cam=all_grayscale_cam)
    #
    # # find the top 30 feature
    # index = np.argsort(all_grayscale_cam.ravel())[:-21:-1]
    # pos = np.unravel_index(index, all_grayscale_cam.shape)
    # pos = np.column_stack(pos)
    # print('the top 30 important feature is:')
    # print(pos)
    #
    # # todo: set the threshold
    # threshold = args.threshold
    # print('the important feature num is:{}'.format(np.sum(all_grayscale_cam >= threshold)))
    # print('the feature index:(according to the axi0)')
    # indexes = np.argwhere(all_grayscale_cam >= threshold)
    # print(indexes + 1)
    # indexes_col = indexes[np.lexsort(indexes.T)]
    # print('the feature index:(according to the axi1)')
    # print(indexes_col + 1)
    # all_grayscale_cam[all_grayscale_cam >= threshold] = 1
    # all_grayscale_cam[all_grayscale_cam <= threshold] = 0
    #
    # cam_image = show_cam_on_image(input_numpy, all_grayscale_cam)
    # cv2.imwrite(f'PVT_{args.method}_' + args.subjects + '.jpg', cam_image)
    # np.savez(f'PVT_{args.method}_' + args.subjects + '.npz',
    #          cam1=all_grayscale_cam_no_threshold,
    #          cam2=all_grayscale_cam,
    #          top20=pos,
    #          indexes=indexes,
    #          indexes_col=indexes_col)
