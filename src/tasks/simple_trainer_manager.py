import os
import sys
import time
import math

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from utils.tools import AverageMeter, EarlyStopping
from utils.logger import Logger
from utils.statistics import Statistics
from utils.messages import Messages
from metrics.metrics import compute_accuracy, compute_confusion_matrix, extract_stats_from_confm
from tensorboardX import SummaryWriter


class SimpleTrainer(object):
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model
        self.logger_stats = Logger(cf.log_file_stats)
        self.stats = Statistics()
        self.msg = Messages()
        self.validator = self.validation(self.logger_stats, self.model, cf, self.stats, self.msg)
        self.trainer = self.train(self.logger_stats, self.model, cf, self.validator, self.stats, self.msg)
        self.predictor = self.predict(self.logger_stats, self.model, cf)

    class train(object):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            # Initialize training variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.validator = validator
            self.logger_stats.write('\n- Starting train <--- \n')
            self.curr_epoch = 1 if self.model.best_stats.epoch == 0 else self.model.best_stats.epoch
            self.stop = False
            self.stats = stats
            self.best_acc = 0
            self.msg = msg
            self.loss = None
            self.outputs = None
            self.labels = None
            self.writer = SummaryWriter(os.path.join(cf.tensorboard_path, 'train'))

        def start(self, train_loader, train_set, valid_set=None, valid_loader=None):
            self.train_num_batches = math.ceil(train_set.num_images / float(self.cf.train_batch_size))
            self.val_num_batches = 0 if valid_set is None else math.ceil(valid_set.num_images / float(self.cf.valid_batch_size))

            # Define early stopping control
            if self.cf.early_stopping:
                early_stopping = EarlyStopping(self.cf)
            else:
                early_stopping = None

            # Train process
            for epoch in tqdm(range(self.curr_epoch, self.cf.epochs + 1), desc='Epochs...', file=sys.stdout):
                # Shuffle train data
                train_set.update_indexes()

                # Initialize logger
                epoch_time = time.time()
                self.logger_stats.write('\t ------ Epoch: ' + str(epoch) + ' ------ \n')

                # Initialize stats
                self.stats.epoch = epoch
                self.train_loss = AverageMeter()
                self.confm_list = np.zeros((self.cf.num_classes, self.cf.num_classes))

                # Train epoch
                self.training_loop(epoch, train_loader)

                # Save stats
                self.stats.train.conf_m = self.confm_list
                self.compute_stats(self.confm_list, self.train_loss)
                self.save_stats_epoch(epoch)
                self.logger_stats.write_stat(self.stats.train, epoch,
                                             os.path.join(self.cf.train_json_path,
                                                          'train_epoch_' + str(epoch) + '.json'))

                # Validate epoch
                self.validate_epoch(valid_set, valid_loader, early_stopping, epoch)

                # Update scheduler
                if self.model.scheduler is not None:
                    self.model.scheduler.step(self.stats.val.loss)

                # Saving model if score improvement
                new_best = self.model.save(self.stats)
                if new_best:
                    self.logger_stats.write_best_stats(self.stats, epoch, self.cf.best_json_file)

                if self.stop:
                    return

            # Save model without training
            if self.cf.epochs == 0:
                self.model.save_model()

        def training_loop(self, epoch, train_loader):
            # Train epoch
            for i, data in tqdm(enumerate(train_loader), desc="Training...", total=len(train_loader), file=sys.stdout):
                # Read Data
                inputs, labels = data

                n, c, w, h = inputs.size()
                inputs = Variable(inputs).cuda()
                self.inputs = inputs
                self.labels = Variable(labels).cuda()

                # Predict model
                self.model.optimizer.zero_grad()
                self.outputs = self.model.net(inputs)
                predictions = self.outputs.data.max(1)[1].cpu().numpy()

                # Compute gradients
                self.compute_gradients()

                # Compute batch stats
                self.train_loss.update(float(self.loss.cpu().item()), n)
                confm = compute_confusion_matrix(predictions, self.labels.cpu().data.numpy(), self.cf.num_classes,
                                                 self.cf.void_class)
                self.confm_list = self.confm_list + confm

                if self.cf.normalize_loss:
                    self.stats.train.loss = self.train_loss.avg
                else:
                    self.stats.train.loss = self.train_loss.avg

                if not self.cf.debug:
                    # Save stats
                    self.save_stats_batch((epoch - 1) * self.train_num_batches + i)

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100. * self.stats.train.acc, epoch)

        def save_stats_batch(self, batch):
            # Save logger
            if batch is not None:
                self.writer.add_scalar('losses/batch', self.stats.train.loss, batch)

        def compute_gradients(self):
            self.loss = self.model.loss(self.outputs, self.labels)
            self.loss.backward()
            self.model.optimizer.step()

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.train.acc = np.nanmean(mean_accuracy)
            self.stats.train.loss = float(train_loss.avg.cpu().data)

        def validate_epoch(self, valid_set, valid_loader, early_Stopping, epoch):
            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                         self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop = True
                # Set model in training mode
                self.model.net.train()

    class validation(object):
        def __init__(self, logger_stats, model, cf, stats, msg):
            # Initialize validation variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.stats = stats
            self.msg = msg
            self.writer = SummaryWriter(os.path.join(cf.tensorboard_path, 'validation'))

        def start(self, valid_set, valid_loader, mode='Validation', epoch=None, save_folder=None):
            confm_list = np.zeros((self.cf.num_classes, self.cf.num_classes))

            self.val_loss = AverageMeter()

            # Validate model
            if self.cf.problem_type == 'detection':
                self.validation_loop(epoch, valid_loader, valid_set, save_folder)
            else:
                self.validation_loop(epoch, valid_loader, valid_set, confm_list)

            # Compute stats
            self.compute_stats(np.asarray(self.stats.val.conf_m), self.val_loss)

            # Save stats
            self.save_stats(epoch)
            if mode == 'Epoch Validation':
                self.logger_stats.write_stat(self.stats.train, epoch,
                                             os.path.join(self.cf.train_json_path,
                                                          'valid_epoch_' + str(epoch) + '.json'))
            elif mode == 'Validation':
                self.logger_stats.write_stat(self.stats.val, epoch, self.cf.val_json_file)
            elif mode == 'Test':
                self.logger_stats.write_stat(self.stats.test, epoch, self.cf.test_json_file)

        def validation_loop(self, epoch, valid_loader, valid_set, confm_list):
            for vi, data in tqdm(enumerate(valid_loader), desc="Validating...", total=len(valid_loader),
                                 file=sys.stdout):
                # Read data
                inputs, gts = data
                n_images, w, h, c = inputs.size()
                inputs = Variable(inputs).cuda()
                gts = Variable(gts).cuda()

                # Predict model
                with torch.no_grad():
                    outputs = self.model.net(inputs)
                    predictions = outputs.data.max(1)[1].cpu().numpy()

                    # Compute batch stats
                    self.val_loss.update(float(self.model.loss(outputs, gts).cpu().item() / n_images), n_images)
                    confm = compute_confusion_matrix(predictions, gts.cpu().data.numpy(), self.cf.num_classes,
                                                     self.cf.void_class)
                    confm_list = confm_list + confm

                # Save epoch stats
                self.stats.val.conf_m = confm_list
                if not self.cf.normalize_loss:
                    self.stats.val.loss = self.val_loss.avg
                else:
                    self.stats.val.loss = self.val_loss.avg

                # Save predictions and generate overlaping
                self.update_tensorboard(inputs.cpu(), gts.cpu(),
                                        predictions, epoch, range(vi * self.cf.valid_batch_size,
                                                                  vi * self.cf.valid_batch_size +
                                                                  np.shape(predictions)[0]),
                                        valid_set.num_images)

        def update_tensorboard(self, inputs, gts, predictions, epoch, indexes, val_len):
            pass

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f] \n' % (
                    epoch, self.stats.val.loss, 100 * self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')
            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[val loss %.5f], [acc %.2f] \n' % (
                    self.stats.val.loss, 100 * self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')

    class predict(object):
        def __init__(self, logger_stats, model, cf):
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf

        def start(self, dataloader):
            self.model.net.eval()

            for vi, data in tqdm(enumerate(dataloader), desc='Predicting...', total=len(dataloader), file=sys.stdout):
                inputs, img_name, img_shape = data

                inputs = Variable(inputs).cuda()
                with torch.no_grad():
                    outputs = self.model.net(inputs)
                    predictions = outputs.data.max(1)[1].cpu().numpy()

                    self.write_results(predictions, img_name, img_shape)

        def write_results(self, predictions, img_name, img_shape):
            pass
