import os

import numpy as np

from .simple_trainer_manager import SimpleTrainer
from metrics.metrics import compute_precision, compute_recall, compute_f1score, compute_accuracy, extract_stats_from_confm
from utils.tools import confm_metrics2image


class ClassificationManager(SimpleTrainer):
    def __init__(self, cf, model):
        super(ClassificationManager, self).__init__(cf, model)

    class train(SimpleTrainer.train):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            super(ClassificationManager.train, self).__init__(logger_stats, model, cf, validator, stats, msg)
            if self.cf.resume_experiment:
                self.msg.msg_stats_best = 'Best case: epoch = %d, acc= %.2f, precision= %.2f, recall= %.2f, ' \
                                          'f1score= %.2f, loss = %.5f\n' % (self.model.best_stats.epoch,
                                                                            100 * self.model.best_stats.val.acc,
                                                                            100 * self.model.best_stats.val.precision,
                                                                            100 * self.model.best_stats.val.recall,
                                                                            100 * self.model.best_stats.val.f1score,
                                                                            self.model.best_stats.val.loss)

        def validate_epoch(self, valid_set, valid_loader, early_stopping, epoch):
            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch)

                # Early stopping checking
                if self.cf.early_stopping:
                    if early_stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                            self.stats.val.acc, self.stats.val.f1score):
                        self.stop = True

                # Set model in training mode
                self.model.net.train()

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)

            tp = np.sum(TP_list)
            tn = np.sum(TN_list)
            fp = np.sum(FP_list)
            fn = np.sum(FN_list)

            r = tp / (tp + fn)
            p = tp / (tp + fp)

            self.stats.train.acc = (tp + tn) / (tp + tn + fp + fn)
            self.stats.train.recall = r
            self.stats.train.precision = p
            self.stats.train.f1score = 2 * (r * p) / (r + p)
            if train_loss is not None:
                self.stats.train.loss = train_loss.avg

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                # self.writer.add_scalar('metrics/accuracy', 100. * self.stats.train.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100. * self.stats.train.precision, epoch)
                # self.writer.add_scalar('metrics/recall', 100. * self.stats.train.recall, epoch)
                # self.writer.add_scalar('metrics/f1score', 100. * self.stats.train.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.train.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')

                # Save learning rate
                # self.logger_stats.write(str(self.model.scheduler.get_lr()))    # Step, MultiStep
                for param_group in self.model.optimizer.param_groups:
                    self.writer.add_scalar('lr/lr', param_group['lr'], epoch)  # ReduceLROnPlateau
                # self.logger_stats.write(str(param_group['lr']))

    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats, msg):
            super(ClassificationManager.validation, self).__init__(logger_stats, model, cf, stats, msg)

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            tp = np.sum(TP_list)
            tn = np.sum(TN_list)
            fp = np.sum(FP_list)
            fn = np.sum(FN_list)

            r = tp / (tp + fn)
            p = tp / (tp + fp)

            self.stats.val.acc = (tp + tn) / (tp + tn + fp + fn)
            self.stats.val.recall = r
            self.stats.val.precision = p
            self.stats.val.f1score = 2 * (r * p) / (r + p)
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch, mode):
            # Save logger
            if epoch is not None:
                # add scores to log
                self.logger_stats.write('Classification validation scores:\n')
                self.logger_stats.write(
                    '[epoch %d], [val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        epoch, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))

                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.val.loss, epoch)
                # self.writer.add_scalar('metrics/accuracy', 100. * self.stats.val.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100. * self.stats.val.precision, epoch)
                # self.writer.add_scalar('metrics/recall', 100. * self.stats.val.recall, epoch)
                # self.writer.add_scalar('metrics/f1score', 100. * self.stats.val.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.val.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')
            else:
                self.logger_stats.write('----------------- Scores summary --------------------\n')
                self.logger_stats.write(
                    '[%s loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        mode, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')

    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(ClassificationManager.predict, self).__init__(logger_stats, model, cf)
            self.filename = os.path.join(self.cf.predict_path_output, 'predictions.txt')
            self.f = open(self.filename, 'w')

        def write_results(self, predictions, img_name, img_shape):
            msg = img_name[0] + ' ' + str(predictions[0]) + '\n'
            self.f.writelines(msg)
