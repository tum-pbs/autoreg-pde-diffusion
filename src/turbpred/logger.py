import logging
import os, sys
import shutil
from datetime import datetime
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class Logger(object):
    def __init__(self, path:str, override:bool=False, addNumber:bool=True, addDate:bool=False):
        if addDate:
            self.path = "runs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + path
        elif addNumber:
            self.path = "runs/%s_%02d" % (path, 0)
        else:
            self.path = "runs/" + path

        if os.path.isdir(self.path):
            if override:
                shutil.rmtree(self.path)
            else:
                if addNumber:
                    num = 1
                    while os.path.isdir(self.path):
                        self.path = "runs/%s_%02d" % (path, num)
                        num += 1

        os.makedirs(self.path)

        shutil.copy(sys.argv[0], os.path.join(self.path, "training.py"))

        self.tfWriter = CustomSummaryWriter(self.path, flush_secs=20)

        # hacky reload fix for logging to work properly
        import importlib
        importlib.reload(logging)
        logging.basicConfig(filename=self.path+"/log.txt", format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
        logging.info("Path: %s" % self.path)
        logging.info("PyTorch Seed: %d" % torch.random.initial_seed())



    def setup(self, model:Module, optimizer:Optimizer):
        self.model = model
        self.optimizer = optimizer

        layout = {}

        layout["Training"] = {
            "Loss":                                  ["Multiline", ["_train/epoch_lossFull",]],
            "Loss Reconstruction":                   ["Multiline", ["_train/epoch_lossRec"]],
            "Loss Prediction":                       ["Multiline", ["_train/epoch_lossPred"]],
            "Comparison Rec vs 1. Pred Error":       ["Multiline", ["_train/epoch_compRec1.Pred"]],
            "Comparison 1. Pred vs Last Pred Error": ["Multiline", ["_train/epoch_comp1.LastPred"]],
        }
        testNames = [
                    ("Test Extrapolate Mach 0.50-0.52", "extrap"),
                    ("Test Interpolate Mach 0.66-0.68", "interp"),
                    ("Test Longer Rollout Mach 0.64-0.65", "longer"),
                    ("Test Low Reynolds 100-200", "lowRey"),
                    ("Test High Reynolds 900-1000", "highRey"),
                    ("Test Varying Reynolds Number (200-900)", "varReyIn"),
                    ("Test Z Slice 200-300", "zInterp"),]
        for names in testNames:
            layout[names[0]] = {
                "MSE Reconstruction":                    ["Multiline", ["%s/epoch_lossRecMSE"     % (names[1])]],
                "MSE Prediction":                        ["Multiline", ["%s/epoch_lossPredMSE"    % (names[1])]],
                "LSIM Reconstruction":                   ["Multiline", ["%s/epoch_lossRecLSIM"    % (names[1])]],
                "LSIM Prediction":                       ["Multiline", ["%s/epoch_lossPredLSIM"   % (names[1])]],
                "Comparison Rec vs 1. Pred Error":       ["Multiline", ["%s/epoch_compRec1.Pred"  % (names[1])]],
                "Comparison 1. Pred vs Last Pred Error": ["Multiline", ["%s/epoch_comp1.LastPred" % (names[1])]],
            }

        self.tfWriter.add_custom_scalars(layout)

    def close(self):
        logging.info("\nLog completed.")
        logging.shutdown()
        self.tfWriter.close()


    def saveTrainState(self, epoch:int, checkpointEvery:int=200):
        assert (self.model), "No model to save, setup logger first!"

        torch.save(self.optimizer.state_dict, self.path + "/TrainState.pth")
        self.model.save(self.path, -1, noPrint=True)

        if epoch % checkpointEvery == 0:
            self.model.save(self.path, epoch, noPrint=True)




# Adjust hParam behavior of SummaryWriter to store results in a single folder
# Workaround from:
# https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
class CustomSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        # remove all lists from hParam dict since only int, float, str, bool and torch.Tensor are possible
        for key, value in hparam_dict.items():
            if type(value) is list:
                valueStr = " ".join([str(elem) for elem in value])
                hparam_dict[key] = valueStr
            elif not type(value) in [int, float, str, bool, torch.Tensor]:
                hparam_dict[key] = " "

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        w_hp = self._get_file_writer()
        w_hp.add_summary(exp)
        w_hp.add_summary(ssi)
        w_hp.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)