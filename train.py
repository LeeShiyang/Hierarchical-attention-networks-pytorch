"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epoches", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--word_feature_size", type=int, default=4)
parser.add_argument("--sent_feature_size", type=int, default=3)
parser.add_argument("--num_bins", type=int, default=10)
parser.add_argument("--es_min_delta", type=float, default=0.0,
                    help="Early stopping's parameter: minimum change loss to qualify as an improvement")
parser.add_argument("--es_patience", type=int, default=5,
                    help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
parser.add_argument("--train_data", type=str, default="/disk/home/klee/data/cs_merged_tokenized_superspan_HANs.txt")
parser.add_argument("--train_label", type=str, default="/disk/home/klee/data/cs_merged_label")
parser.add_argument("--test_data", type=str, default="/disk/home/klee/data/cs_merged_tokenized_superspan_HANs.txt")
parser.add_argument("--test_label", type=str, default="/disk/home/klee/data/cs_merged_label")
parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
parser.add_argument("--dict", type=str, default="/disk/home/klee/data/cs_merged_tokenized_dictionary.bin")
parser.add_argument("--feature_path", type=str, default="/disk/home/klee/data/cs_merged_tokenized_concept_feature.bin")
parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
parser.add_argument("--saved_path", type=str, default="trained_models")
args = parser.parse_args()
use_cuda = True

tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)
supersequence_path = tokenized_text + '_superspan_sequence.json'
superspan_HANsFile = tokenized_text + '_superspan_HANs.txt'
phrases2feature_vector_path = tokenized_text + '_phrases2feature_vector.bin'
ImportanceFeatureMatsFile = tokenized_text + '_ImportanceFeatureMatsFile.bin'
model_save_path = supersequence_path + '_embedding.bin'
Vv_embedding_path = tokenized_text + '_Vv_embedding.bin'
path_semanticsFile = tokenized_text + '_basic_semantics.bin'
class_idsFile = tokenized_text + '_class_ids.bin'
VvFile = tokenized_text + '_Vv.bin'
max_vocab = 500000

if use_cuda:
    torch.cuda.set_device(1)


def train(opt):
    if use_cuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    training_set = MyDataset(opt.train_data, opt.train_label, opt.dict, ImportanceFeatureMatsFile, max_vocab, class_idsFile, VvFile, model_save_path)
    training_generator = DataLoader(training_set, **training_params)
    test_set = training_set  # MyDataset(opt.test_data,opt.test_label ,opt.word2vec_path)
    test_generator = training_generator  # DataLoader(test_set, **test_params)

    model = HierAttNet(opt.sent_feature_size, phrases2feature_vector_path, opt.dict,
                       training_set.max_length_sentences, training_set.max_length_word,
                       model_save_path, Vv_embedding_path, path_semanticsFile, max_vocab, use_cuda, training_set, opt.num_bins)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, (feature, ImportanceFeatureMat, label, text) in enumerate(training_generator):
            if use_cuda:
                feature = feature.cuda()
                ImportanceFeatureMat = ImportanceFeatureMat.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            predictions, attn_score = model(feature, ImportanceFeatureMat, label)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy", "top K accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, top K accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["top K accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if use_cuda:
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    te_predictions, te_attn_score = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


train(args)
