import torch
from data_utils import get_dataset, get_dataloader
from model import SCDC
import argparse
from evaluate import inference, evaluate
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='MouseHSPC',
                        help="dataset",
                        choices=['MouseHSPC', 'CellLine'],
                        type=str)
    parser.add_argument("--batch_size",
                        help="batch size",
                        default=256,
                        type=int)
    parser.add_argument("--gamma",
                        help="the variance of Gaussian noise",
                        default=2.5,
                        type=float)
    parser.add_argument("--train_epoch",
                        help="training epochs",
                        default=300,
                        type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    args = parser.parse_args()
    print('=======Arguments=======')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # prepare data
    data, size_factors, raw_counts, cell_types, batch_indices, n_cell, n_gene, n_type, n_batch, \
    cell_type_dict, batch_dict, gene_names = get_dataset(data=args.dataset)
    train_dataloader, test_dataloader = get_dataloader(
        data,
        size_factors,
        raw_counts,
        cell_types,
        batch_indices,
        batch_size=args.batch_size)

    # build model
    net = SCDC(n_cell=n_cell, n_gene=n_gene, n_type=n_type,
                n_batch=n_batch).cuda()

    # training
    print('=======Start=======')
    net.run(epochs=args.train_epoch,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            gamma=args.gamma)
    print('=======Finish=======\n')

    # evaluation
    print('=======Analyzing Results=======')
    feature_vec, type_vec, batch_vec, pred_vec, soft_vec = inference(
        net, test_dataloader)
    evaluate(feature_vec,
             pred_vec,
             type_vec,
             batch_vec,
             soft_vec,
             batch_metric=True)
    print('=======Finish=======\n')