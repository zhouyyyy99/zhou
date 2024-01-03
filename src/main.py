
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json
import numpy as np
import matplotlib as plt

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# i add later
import sklearn.metrics as results

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'],
                         limit_size=config['limit_size'], length_size=config['length_size'],
                         dim_size=config['dim_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features

    # print('====================== feature dim : ==============================')
    # print(my_data.feature_df)

    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config[
        'test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    # print("val_ratio = ", config['val_ratio'])

    # print('-------- the number of all_IDs -----------')
    # print(len(my_data.all_IDs))

    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(config=config,
                                                                 data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],
                                                                 # used only if test_indices not explicitly specified
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        # print('-------------------- Pre-process features ----------------------')
        # print(my_data.feature_df)

        if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # 双样本 t 检验进行特征筛选（功能连接 flatten 矩阵）
    # 这里得到train下标、val下表、test下标
    # 通过train的下标，在文件里找到train的数据，进行双样本t检验，得到feature的下标
    if config['feature_t_test']:
        if config['mdd_data_type'] == 'corr':
            print("t_test begin")
            corr_file = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_corr_up_flatten.npz'
            feature_mask_index = utils.t_test_selected(p=0.005, data_filename=corr_file, train_indices=train_indices)
            print("t_test over")
    else:
        feature_mask_index = None

    # Create model
    logger.info("Creating model ...")
    print(my_data)
    model = model_factory(config, my_data)

    if config['freeze']:
        print("freezed model")
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices, feature_mask_index)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices, feature_mask_index)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices, feature_mask_index)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config[
                             'key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    val_loss_list = []  # store val loss
    val_acc_list = []  # store val acc

    # store the top 5 acc best metrics
    top_5_metrics = []

    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    val_loss_list.append(aggr_metrics_val['loss'])
    if (config['task'] is not 'imputation'):
        val_acc_list.append(aggr_metrics_val['accuracy'])

    logger.info('Starting training...')
    train_epoch_loss_list = []  # store the loss of each epoch
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():  # k:key  v:value
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        train_epoch_loss_list.append(aggr_metrics_train['loss'])

        train_loss_save_path = os.path.join(config['output_dir'], 'train_loss.jpg')
        utils.plot_loss(train_epoch_loss_list, train_loss_save_path)

        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info(
            "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

            val_loss_list.append(aggr_metrics_val['loss'])

            if (config['task'] is not 'imputation'):
                val_acc_list.append(aggr_metrics_val['accuracy'])
                utils.plot_loss(val_acc_list, save_path=os.path.join(config['output_dir'], 'val_acc.jpg'),
                                label_name='val_acc')

            utils.plot_loss(val_loss_list, os.path.join(config['output_dir'], 'val_loss.jpg'),
                            label_name='val_loss')



        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    print('the train epoch loss list length is :{}'.format(len(train_epoch_loss_list)))

    return best_metrics, top_5_metrics


if __name__ == '__main__':

    k = 5
    k_fold_acc_list = []
    k_fold_sen_list = []
    k_fold_spe_list = []
    k_fold_auroc_list = []
    k_fold_auprc_list = []


    best_results = []

    for i in range(k):

        args = Options().parse()  # `argsparse` object
        config = setup(args, i)  # configuration dictionary

        # main(config)
        try:
            best_metrics, top_5_metrics = main(config)
        except RuntimeError as exception:
            if "memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if (config['task'] is not 'imputation'):
            k_fold_acc_list.append(best_metrics['accuracy'])
        k_fold_sen_list.append(best_metrics['SEN'])
        k_fold_spe_list.append(best_metrics['SPE'])
        k_fold_auroc_list.append(best_metrics['AUROC'])
        k_fold_auprc_list.append((best_metrics['AUPRC']))

        # file = open(os.path.join(config['output_dir'], 'metrics_result.txt'), 'w')
        # for item in top_5_metrics:
        #     item = json.dumps(item)
        #     file.write(item)
        #     file.write('\n')
        # file.close()

        best_results.append(best_metrics)

    print('---------------------------------------------------------')
    print('all of the results:')
    print('acc:{}'.format(str(k_fold_acc_list)))
    print('sen:{}'.format(str(k_fold_sen_list)))
    print('spe:{}'.format(str(k_fold_spe_list)))
    print('auroc:{}'.format(str(k_fold_auroc_list)))
    print('auprc:{}'.format(str(k_fold_auprc_list)))

    print('---------------------------------------------------------')
    print('all best metrics:')
    for item in best_results:
        print(item)

    print('---------------------------------------------------------')
    print('acc:{}'.format(np.mean(k_fold_acc_list)))
    print('sen:{}'.format(np.mean(k_fold_sen_list)))
    print('spe:{}'.format(np.mean(k_fold_spe_list)))
    print('auroc:{}'.format(np.mean(k_fold_auroc_list)))
    print('sauprc:{}'.format(np.mean(k_fold_auprc_list)))
