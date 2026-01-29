import copy
import math
import sys
import random


import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop

import torch.nn.functional as F
import torch.nn.init as init


def random_reinitialize_layers(
    model, fraction=0.1, exclude_names=("fc", "classifier", "head")
):
    """
    Randomly reinitializes a fraction of the layers in the model.
    Excludes layers by name (e.g., final classification head).
    """
    layers = [
        m for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    layers = [
        l
        for l in layers
        if not any(name in l.__class__.__name__.lower() for name in exclude_names)
    ]

    n_reset = max(1, int(len(layers) * fraction))
    to_reset = random.sample(layers, n_reset)

    for layer in to_reset:
        if isinstance(layer, torch.nn.Linear):
            init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(layer.bias, -bound, bound)
        elif isinstance(layer, torch.nn.Conv2d):
            init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                init.constant_(layer.bias, 0.0)
    print(f"[Reinit] Reinitialized {n_reset} layers.")


class RandomResetOnOverfit:
    def __init__(
        self,
        model,
        train_criterion,
        noise_level,
        margin=0.02,
        final_margin=0.02,
        check_every=5,
        reset_fraction=0.1,
        margin_update=0.5,
        num_update=3,
    ):
        self.model = model
        self.train_criterion = train_criterion
        self.target_acc = 1.0 - noise_level
        self.margin = margin
        self.final_margin = final_margin
        self.margin_update = margin_update
        self.margin_update = np.linspace(
            self.margin, self.final_margin, endpoint=True, num=num_update + 1
        ).tolist()[1:-1]
        self.margin_update.append(2)
        self.margin_update = self.margin_update[::-1]

        self.check_every = check_every
        self.reset_fraction = reset_fraction
        self.epoch = 0

    def maybe_reset(self, train_acc):
        self.epoch += 1
        # if self.epoch % self.check_every != 0:
        #     return
        if train_acc > self.target_acc + self.margin:
            print(
                f"[Reinit] Accuracy {train_acc:.4f} > {self.target_acc + self.margin:.4f}, resetting."
            )
            random_reinitialize_layers(self.model, fraction=self.reset_fraction)
            if hasattr(self.train_criterion, "soft_threshold"):
                print("Parameter reinit")
                # maybe it would be better to perform a soft thresholding
                # (and a masked reinit )
                self.train_criterion.soft_threshold(self.target_acc)
            # self.margin += self.margin_update * (self.final_margin - self.margin)
            self.margin = self.margin_update.pop()
        else:
            print(
                f"[NoReinit] Accuracy {train_acc:.4f} < {self.target_acc + self.margin:.4f}."
            )


class AdaptiveRegularizer:
    def __init__(
        self,
        optimizer,
        noise_level,
        wd_init=1e-3,
        wd_min=1e-4,
        wd_max=1e-1,
        beta_down=0.9,
        beta_up=1.1,
        progress_threshold=0.01,
    ):
        self.optimizer = optimizer
        self.a_star = 1.0 - noise_level
        self.wd = wd_init
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.beta_down = beta_down
        self.beta_up = beta_up
        self.progress_threshold = progress_threshold
        self.prev_acc = None

    def update(self, acc):
        if self.prev_acc is None:
            self.prev_acc = acc
            return

        delta_acc = acc - self.prev_acc

        if acc < self.a_star and delta_acc < self.progress_threshold:
            # Learning stalled below target, reduce regularization
            self.wd = max(self.wd_min, self.wd * self.beta_down)
        elif acc >= self.a_star or delta_acc >= self.progress_threshold:
            # Learning well or above target, increase regularization
            self.wd = min(self.wd_max, self.wd * self.beta_up)
        # else: keep weight decay unchanged

        # Apply weight decay to optimizer
        for group in self.optimizer.param_groups:
            group["weight_decay"] = self.wd

        print(
            f"Adaptive WD: {self.wd:.6f}, Acc: {acc:.4f}, ΔAcc: {delta_acc:.4f}, Target: {self.a_star:.4f}"
        )

        self.prev_acc = acc


class AsymmetricWeightDecay:
    def __init__(
        self,
        optimizer,
        noise_level,
        # wd_base=0.001,
        # wd_min=1e-5,
        # wd_max=1e-1,
        wd_base=0.01,
        wd_min=1e-3,
        wd_max=1e-1,
        delta=0.05,
        gamma=2.0,
    ):
        self.optimizer = optimizer
        self.a_star = 1.0 - noise_level
        self.wd_base = wd_base
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.delta = delta
        self.gamma = gamma

    def update(self, acc):
        if acc < self.a_star:
            factor = 1.0 - self.gamma * (self.a_star - acc) / self.delta
        else:
            factor = 1.0 + (acc - self.a_star) / self.delta

        wd = self.wd_base * factor
        wd = max(self.wd_min, min(self.wd_max, wd))

        for g in self.optimizer.param_groups:
            g["weight_decay"] = wd

        print(f"[AsymWD] acc={acc:.4f}, wd={wd:.6f}, target={self.a_star:.4f}")


def create_ema_model(model, decay=0.999):
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    def update_ema(original_model):
        with torch.no_grad():
            for ema_param, param in zip(
                ema_model.parameters(), original_model.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    return ema_model, update_ema


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        reparametrization_net,
        train_criterion,
        metrics,
        optimizer,
        optimizer_loss,
        config,
        data_loader,
        valid_data_loader=None,
        test_data_loader=None,
        lr_scheduler=None,
        lr_scheduler_overparametrization=None,
        len_epoch=None,
        val_criterion=None,
    ):
        super().__init__(
            model,
            reparametrization_net,
            train_criterion,
            metrics,
            optimizer,
            optimizer_loss,
            config,
            val_criterion,
        )
        self.clip_val = 1
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_overparametrization = lr_scheduler_overparametrization
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        self.ema_decay = 0.99

        # Usage:
        self.ema_model, self.ema_update_fn = create_ema_model(
            self.model, self.ema_decay
        )
        self.ema_model.to(self.device)

        self.train_criterion = train_criterion

        self.new_best_val = False
        self.val_acc = 0
        self.test_val_acc = 0

        self.best_score = {
            "test": 0,
            "train": 0,
            "epoch": 0,
        }
        #

        # self.weight_decay = AsymmetricWeightDecay(
        #     self.optimizer, noise_level=0.4020799994468689
        # )

        self.random_reset = RandomResetOnOverfit(
            self.model,
            self.train_criterion,
            noise_level=0.4020799994468689,
            margin=-0.1,
            # margin=0.001,
        )
        # self.weight_decay = AdaptiveRegularizer(
        #     self.optimizer, noise_level=0.4020799994468689
        # )
        self.old_train_accuracy = 0.0
        self.offset = -0.02

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            if self.writer is not None:
                self.writer.add_scalar({"{}".format(metric.__name__): acc_metrics[i]})
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        noise_level = 0

        if hasattr(self.train_criterion, "set_epoch"):
            self.train_criterion.set_epoch(epoch)

        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, data2, label, indexs, *_) in enumerate(progress):
                progress.set_description_str(f"Train epoch {epoch}")

                data, label = data.to(self.device), label.long().to(self.device)

                target = (
                    torch.zeros(len(label), self.config["num_classes"])
                    .to(self.device)
                    .scatter_(1, label.view(-1, 1), 1)
                )

                if self.config["train_loss"]["args"]["ratio_consistency"] > 0:
                    data2 = data2.to(self.device)
                    data_all = torch.cat([data, data2]).cuda()
                else:
                    data_all = data

                output = self.model(data_all)
                ema_output = None
                # # Use ema_model directly for inference
                # with torch.no_grad():
                #     ema_output = self.ema_model(data_all)

                loss = self.train_criterion(
                    indexs, output, label=target  # , ema_output=ema_output
                )

                if self.optimizer_loss is not None:
                    self.optimizer_loss.zero_grad()

                if self.optimizer is not None:
                    self.optimizer.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     self.model.parameters(), self.clip_val
                # )
                # if hasattr(self.train_criterion, "u") and hasattr(
                #     self.train_criterion, "v"
                # ):
                #     torch.nn.utils.clip_grad_norm_(
                #         [self.train_criterion.u, self.train_criterion.v],
                #         self.clip_val,
                #     )

                if self.optimizer_loss is not None:
                    self.optimizer_loss.step()
                if self.optimizer is not None:
                    self.optimizer.step()
                self.ema_update_fn(self.model)

                if self.config["train_loss"]["args"]["ratio_consistency"] > 0:
                    output, _ = torch.chunk(output, 2)

                if self.writer is not None:

                    self.writer.set_step(
                        (epoch - 1) * self.len_epoch + batch_idx, epoch=epoch
                    )
                    self.writer.add_scalar({"loss": loss.item()})

                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(
                            self._progress(batch_idx), loss.item()
                        )
                    )

                if batch_idx == self.len_epoch:
                    break

        log = {
            "loss": total_loss / self.len_epoch,
            "noise level": noise_level / self.len_epoch,
            "metrics": (total_metrics / self.len_epoch).tolist(),
            "learning rate": self.lr_scheduler.get_lr(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

        test_acc = test_log["test_metrics"][0]
        if test_acc > self.best_score["test"]:
            self.best_score["test"] = test_acc
            self.best_score["train"] = log["metrics"][0]
            self.best_score["epoch"] = epoch

        print(self.best_score)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.lr_scheduler_overparametrization is not None:
            self.lr_scheduler_overparametrization.step()

        # noise_level = 0.4020799994468689
        # train_accurary = log["metrics"][0]
        # target_train_accurary = 1 - noise_level

        # lr_min = 0.02
        # lr_max = 50
        # step = 0.05
        # bound_min = np.log10(lr_min)
        # bound_max = np.log10(lr_max)
        # center = target_train_accurary - step
        # beta = 4 / step
        # lr = (bound_max - bound_min) / (
        #     1 + np.exp(-(train_accurary - center) * beta)
        # ) + bound_min
        # lr = 10**lr

        # weight_decay_min = 0.00001
        # weight_decay_max = 0.1
        # step = 0.05
        # bound_min = np.log10(weight_decay_min)
        # bound_max = np.log10(weight_decay_max)
        # center = target_accurary - step
        # beta = 2 / step
        # weight_decay = (bound_max - bound_min) / (
        #     1 + np.exp(-(train_accurary - center) * beta)
        # ) + bound_min
        # weight_decay = 10**weight_decay
        # train_accurary = log["metrics"][0]
        # self.weight_decay.update(train_accurary)

        # self.random_reset.maybe_reset(train_acc=log["metrics"][0])

        # noise_level = 0.4020799994468689
        #
        # target_train_accurary = 1 - noise_level

        # old_train_accuracy = self.old_train_accuracy
        # self.old_train_accuracy = train_accurary

        # weight_decay_min = 0.00001
        # weight_decay_max = 1
        # weightd_decay_at_target = 0.001
        # bound_min = np.log10(weight_decay_min)
        # bound_max = np.log10(weight_decay_max)
        # step = 0.01
        # offset_step = 0.005
        # # pb de la comparaison, si la progression est infinitesimal par rapport à l'ecart entre train_accurary et target_accurary
        # # i.e. c'est dans le bon sens mais cela n'évolue pas assez vite
        # if (train_accurary > target_train_accurary) and (
        #     train_accurary - old_train_accuracy
        #     > 0.05 * (train_accurary - target_train_accurary)
        # ):
        #     self.offset -= offset_step
        # elif (train_accurary < target_train_accurary) and (
        #     train_accurary - old_train_accuracy
        #     < 0.05 * (target_train_accurary - train_accurary)
        # ):
        #     self.offset += offset_step

        # # idea if accuracy is not increasing and accuracy is lower than target accuracy, shift increase offset to decrease the weight decay
        # # if accuracy is above the target accuracy and is not decreasing, lower the offset so as to increase the weight decay

        # center = target_train_accurary + self.offset
        # weight_decay = (weightd_decay_at_target / 4) + (
        #     3 * weightd_decay_at_target / 2
        # ) * np.exp((train_accurary - center) / step)
        # weight_decay = np.minimum(weight_decay_max, weight_decay)
        # weight_decay = np.maximum(weight_decay_min, weight_decay)

        # print(
        #     f"Adaptive weight_decay: {weight_decay}, offset {self.offset}, target accuracy {target_train_accurary}"
        # )

        # if self.optimizer is not None:
        #     for g in self.optimizer.param_groups:
        #         g["weight_decay"] = weight_decay
        # beta = 2 / step
        # weight_decay = (bound_max - bound_min) / (
        #     1 + np.exp(-(train_accurary - center) * beta)
        # ) + bound_min
        # weight_decay = 10**weight_decay

        # alpha =
        # if self.optimizer is not None:
        #     for g in self.optimizer.param_groups:
        #         g["lr"] = 0.001
        # if self.optimizer_loss is not None:
        #     for g in self.optimizer_loss.param_groups:
        #         g["lr"] = lr
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label, indexs, *_) in enumerate(progress):
                    progress.set_description_str(f"Valid epoch {epoch}")
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(
                            output, indexs
                        )
                    loss = self.val_criterion(output, label)

                    if self.writer is not None:
                        self.writer.set_step(
                            (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                            epoch=epoch,
                            mode="valid",
                        )
                        self.writer.add_scalar({"loss": loss.item()})
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        val_acc = (total_val_metrics / len(self.valid_data_loader)).tolist()[0]
        if val_acc > self.val_acc:
            self.val_acc = val_acc
            self.new_best_val = True
            if self.writer is not None:
                self.writer.add_scalar({"Best val acc": self.val_acc}, epoch=epoch)
        else:
            self.new_best_val = False

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        results = np.zeros(
            (len(self.test_data_loader.dataset), self.config["num_classes"]),
            dtype=np.float32,
        )
        tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label, indexs, _) in enumerate(progress):
                    progress.set_description_str(f"Test epoch {epoch}")
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(
                            output, indexs
                        )
                    loss = self.val_criterion(output, label)
                    if self.writer is not None:
                        self.writer.set_step(
                            (epoch - 1) * len(self.test_data_loader) + batch_idx,
                            epoch=epoch,
                            mode="test",
                        )
                        self.writer.add_scalar({"loss": loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    results[indexs.cpu().detach().numpy().tolist()] = (
                        output.cpu().detach().numpy().tolist()
                    )
                    tar_[indexs.cpu().detach().numpy().tolist()] = (
                        label.cpu().detach().numpy().tolist()
                    )

        # add histogram of model parameters to the tensorboard
        top_1_acc = (total_test_metrics / len(self.test_data_loader)).tolist()[0]
        if self.new_best_val:
            self.test_val_acc = top_1_acc
            if self.writer is not None:
                self.writer.add_scalar(
                    {"Test acc with best val": top_1_acc}, epoch=epoch
                )
        if self.writer is not None:
            self.writer.add_scalar({"Top-1": top_1_acc}, epoch=epoch)
            self.writer.add_scalar(
                {
                    "Top-5": (total_test_metrics / len(self.test_data_loader)).tolist()[
                        1
                    ]
                },
                epoch=epoch,
            )

        return {
            "test_loss": total_test_loss / len(self.test_data_loader),
            "test_metrics": (total_test_metrics / len(self.test_data_loader)).tolist(),
        }

    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        data_loader = self.data_loader  # self.loader.run('warmup')

        with tqdm(data_loader) as progress:
            for batch_idx, (data, _, label, indexs, _) in enumerate(progress):
                progress.set_description_str(f"Warm up epoch {epoch}")

                data, label = data.to(self.device), label.long().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                if self.reparametrization_net is not None:
                    output, original_output = self.reparametrization_net(output, indexs)
                out_prob = torch.nn.functional.softmax(output).data.detach()

                loss = torch.nn.functional.cross_entropy(output, label)

                loss.backward()
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.set_step(
                        (epoch - 1) * self.len_epoch + batch_idx, epoch=epoch
                    )
                    self.writer.add_scalar({"loss_record": loss.item()})
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(
                            self._progress(batch_idx), loss.item()
                        )
                    )
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        if hasattr(self.data_loader, "run"):
            self.data_loader.run()
        log = {
            "loss": total_loss / self.len_epoch,
            "metrics": (total_metrics / self.len_epoch).tolist(),
            "learning rate": self.lr_scheduler.get_lr(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

        return log

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
