import os
import time

import numpy as np
import cv2 as cv
from PIL import Image

from .simple_trainer_manager import SimpleTrainer
from metrics.metrics import compute_mIoU, compute_accuracy_segmentation, extract_stats_from_confm
from utils.tools import confm_metrics2image
from utils.save_images import save_img


class SemanticSegmentationManager(SimpleTrainer):
    def __init__(self, cf, model):
        super(SemanticSegmentationManager, self).__init__(cf, model)

    class train(SimpleTrainer.train):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            super(SemanticSegmentationManager.train, self).__init__(logger_stats, model, cf, validator, stats, msg)
            if self.cf.resume_experiment:
                self.msg.msg_stats_best = 'Best case [%s]: epoch = %d, mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
                    self.cf.save_condition, self.model.best_stats.epoch, 100 * self.model.best_stats.val.mIoU,
                    100 * self.model.best_stats.val.acc, self.model.best_stats.val.loss)

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

        def update_messages(self, epoch, epoch_time, new_best):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
                100 * self.stats.val.mIoU, 100 * self.stats.val.acc, self.stats.val.loss)
            if new_best:
                self.msg.msg_stats_best = 'Best case [%s]: epoch = %d, mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
                    self.cf.save_condition, epoch, 100 * self.stats.val.mIoU,
                    100 * self.stats.val.acc, self.stats.val.loss)
                msg_confm = self.stats.val.get_confm_str()
                self.logger_stats.write(msg_confm)
                self.msg.msg_stats_best = self.msg.msg_stats_best + '\nConfusion matrix:\n' + msg_confm

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_IoU = compute_mIoU(TP_list, FP_list, FN_list)
            mean_accuracy = compute_accuracy_segmentation(TP_list, FN_list)
            self.stats.train.acc = np.nanmean(mean_accuracy)
            self.stats.train.mIoU_perclass = mean_IoU
            self.stats.train.mIoU = np.nanmean(mean_IoU)
            if train_loss is not None:
                self.stats.val.loss = train_loss.avg

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100. * self.stats.train.acc, epoch)
                self.writer.add_scalar('metrics/mIoU', 100. * self.stats.train.mIoU, epoch)
                conf_mat_img = confm_metrics2image(self.stats.train.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')

    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats, msg):
            super(SemanticSegmentationManager.validation, self).__init__(logger_stats, model, cf, stats, msg)

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_IoU = compute_mIoU(TP_list, FP_list, FN_list)
            mean_accuracy = compute_accuracy_segmentation(TP_list, FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.mIoU_perclass = mean_IoU
            self.stats.val.mIoU = np.nanmean(mean_IoU)
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch, mode):
            # Save logger
            if epoch is not None:
                # add log
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f], [mean_IoU %.2f] \n' % (
                    epoch, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.mIoU))
                self.logger_stats.write('---------------------------------------------------------------- \n')

                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.val.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100. * self.stats.val.acc, epoch)
                self.writer.add_scalar('metrics/mIoU', 100. * self.stats.val.mIoU, epoch)
                conf_mat_img = confm_metrics2image(self.stats.val.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')
            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[%s loss %.5f], [acc %.2f], [mean_IoU %.2f]\n' % (mode,
                    self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.mIoU))
                self.logger_stats.write('---------------------------------------------------------------- \n')

        def update_tensorboard(self, inputs, gts, predictions, epoch, indexes, val_len):
            if epoch is not None and self.cf.color_map is not None:
                save_img(self.writer, inputs, gts, predictions, epoch, indexes, self.cf.predict_to_save, val_len,
                         self.cf.color_map, self.cf.labels, self.cf.void_class, n_legend_rows=3)

    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(SemanticSegmentationManager.predict, self).__init__(logger_stats, model, cf)

        def write_results(self, predictions, img_name, img_shape):
            path = os.path.join(self.cf.predict_path_output, img_name[0])
            predictions = predictions[0]
            predictions = Image.fromarray(predictions.astype(np.uint8))
            if self.cf.resize_image_test is not None:
                predictions = predictions.resize((img_shape[1],
                                                  img_shape[0]), resample=Image.BILINEAR)
            predictions = np.array(predictions)
            cv.imwrite(path, predictions)
