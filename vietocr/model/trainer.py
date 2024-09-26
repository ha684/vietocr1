import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler

from PIL import Image

from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.tool.translate import build_model, translate, batch_translate_beam_search
from vietocr.tool.utils import download_weights, compute_accuracy
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransformV2
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from vietocr.model.backbone.cnn import CNN


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

import gc
class Trainer:
    def __init__(self, config, pretrained=False, augmentor=ImgAugTransformV2()):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model and move it to the appropriate device
        self.model, self.vocab = build_model(config)


        self.num_iters = config["trainer"]["iters"]
        self.beamsearch = config["predictor"]["beamsearch"]

        self.data_root = config["dataset"]["data_root"]
        self.train_annotation = config["dataset"]["train_annotation"]
        self.valid_annotation = config["dataset"]["valid_annotation"]
        self.train_gen_path = config["dataset"]["train_gen"]
        self.valid_gen_path = config["dataset"]["valid_gen"]
        self.dataset_name = config["dataset"]["name"]
        self.num_epochs = config["trainer"]["epochs"]
        self.batch_size = config["trainer"]["batch_size"]
        self.print_every = config["trainer"]["print_every"]
        self.valid_every = config["trainer"]["valid_every"]
        self.scaler = GradScaler()
        self.image_aug = config["aug"]["image_aug"]
        self.masked_language_model = config["aug"]["masked_language_model"]

        self.checkpoint = config["trainer"]["checkpoint"]
        self.export_weights = config["trainer"]["export"]
        self.metrics = config["trainer"]["metrics"]
        logger = config["trainer"]["log"]
        self.cnn = CNN(config["backbone"], **config["cnn"])
        if logger:
            self.logger = Logger(logger)

        if pretrained:
            weight_file = download_weights(config["pretrain"], quiet=config["quiet"])
            self.load_weights(weight_file)

        self.iter = 0
        self.best_acc = 0
        self.criterion = LabelSmoothingLoss(
            len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1
        )

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.train_gen = self.data_gen(
            os.path.join(self.data_root, self.train_gen_path),
            annotation_path=self.train_annotation,
            masked_language_model=self.masked_language_model,
            transform=transforms,
        )
        self.train_dataset_size = len(self.train_gen.dataset)
        self.iterations_per_epoch = len(self.train_gen)
        total_steps = self.num_epochs * self.iterations_per_epoch

        self.optimizer = AdamW(
            self.model.parameters(), betas=(0.9, 0.98), eps=1e-09
        )
        self.scheduler = OneCycleLR(
            self.optimizer, total_steps=total_steps, **config["optimizer"]
        )

        if self.valid_annotation:
            self.valid_gen = self.data_gen(
                os.path.join(self.data_root, self.valid_gen_path),
                annotation_path=self.valid_annotation,
                masked_language_model=False,
                transform=None,
                is_train=False,
            )

        self.train_losses = []
        self.early_stopping = EarlyStopping(
            patience=config["trainer"].get("patience", 10), verbose=True
        )

    def train(self):
        total_loss = 0
        total_loader_time = 0
        total_gpu_time = 0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()

            for batch in self.train_gen:
                self.iter += 1

                loader_start = time.time()
                total_loader_time += time.time() - loader_start

                gpu_start = time.time()
                loss = self.step(batch)
                total_gpu_time += time.time() - gpu_start
                total_loss += loss
                self.train_losses.append((self.iter, loss))

                if self.iter % self.print_every == 0:
                    avg_loss = total_loss / self.print_every
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    info = (
                        f"Epoch: {epoch}/{self.num_epochs} | "
                        f"Iter: {self.iter} | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Load Time: {total_loader_time:.2f}s | "
                        f"GPU Time: {total_gpu_time:.2f}s"
                    )
                    print(info)
                    if hasattr(self, "logger"):
                        self.logger.log(info)
                    total_loss = 0
                    total_loader_time = 0
                    total_gpu_time = 0

                if self.valid_annotation and self.iter % self.valid_every == 0:
                    val_loss = self.validate()
                    acc_full_seq, acc_per_char = self.precision()
                    torch.cuda.empty_cache()
                    gc.collect()
                    info = (
                        f"Epoch: {epoch}/{self.num_epochs} | "
                        f"Iter: {self.iter} | "
                        f"Valid Loss: {val_loss:.4f} | "
                        f"Acc Full Seq: {acc_full_seq:.4f} | "
                        f"Acc Per Char: {acc_per_char:.4f}"
                    )
                    print(info)
                    if hasattr(self, "logger"):
                        self.logger.log(info)

                    if acc_full_seq > self.best_acc:
                        self.save_weights(self.export_weights)
                        self.best_acc = acc_full_seq
                        info = f"New best accuracy: {self.best_acc:.4f}. Weights saved."
                        print(info)
                        if hasattr(self, "logger"):
                            self.logger.log(info)

                    self.early_stopping(val_loss, self.model, self.export_weights)
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        if hasattr(self, "logger"):
                            self.logger.log("Early stopping triggered.")
                        return

                self.cnn.update_freeze_state()

            epoch_duration = time.time() - epoch_start_time
            info = f"Epoch {epoch} completed in {epoch_duration:.2f}s."
            print(info)
            if hasattr(self, "logger"):
                self.logger.log(info)

            # Save checkpoint at the end of each epoch
            self.save_checkpoint(self.checkpoint)

    def validate(self):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch in self.valid_gen:
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = (
                    batch["img"],
                    batch["tgt_input"],
                    batch["tgt_output"],
                    batch["tgt_padding_mask"],
                )

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(img, tgt_input, tgt_padding_mask)
                    outputs = outputs.flatten(0, 1)
                    tgt_output = tgt_output.flatten()
                    loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())

        total_loss = np.mean(total_loss)
        return total_loss

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_gen:
                batch = self.batch_to_device(batch)

                if self.beamsearch:
                    translated_sentence = batch_translate_beam_search(
                        batch["img"], self.model
                    )
                    prob = None
                else:
                    translated_sentence, prob = translate(batch["img"], self.model)

                pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
                actual_sent = self.vocab.batch_decode(batch["tgt_output"].tolist())

                img_files.extend(batch["filenames"])
                pred_sents.extend(pred_sent)
                actual_sents.extend(actual_sent)

                if prob is not None:
                    all_probs.extend(prob.tolist())
                else:
                    all_probs.extend([None] * len(pred_sent))

                if sample is not None and len(pred_sents) >= sample:
                    pred_sents = pred_sents[:sample]
                    actual_sents = actual_sents[:sample]
                    img_files = img_files[:sample]
                    all_probs = all_probs[:sample]
                    break

        return pred_sents, actual_sents, img_files, all_probs

    def precision(self, sample=None):
        pred_sents, actual_sents, _, _ = self.predict(sample=sample)
        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode="full_sequence")
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode="per_char")
        return acc_full_seq, acc_per_char

    def visualize_prediction(
        self, sample=16, errorcase=False, fontname="serif", fontsize=16
    ):
        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = [
                i for i in range(len(img_files)) if pred_sents[i] != actual_sents[i]
            ]
            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {"family": fontname, "size": fontsize}

        for vis_idx in range(len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx] if probs[vis_idx] is not None else "N/A"

            img = Image.open(img_path)
            plt.figure()
            plt.imshow(img)
            plt.title(
                f"prob: {prob} - pred: {pred_sent} - actual: {actual_sent}",
                loc="left",
                fontdict=fontdict,
            )
            plt.axis("off")

        plt.show()

    def visualize_dataset(self, sample=16, fontname="serif"):
        n = 0
        for batch in self.train_gen:
            imgs = batch["img"]
            tgt_inputs = batch["tgt_input"]
            for i in range(len(imgs)):
                img = imgs[i].cpu().numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(tgt_inputs.T[i].tolist())

                plt.figure()
                plt.title(f"sent: {sent}", loc="center", fontname=fontname)
                plt.imshow(img)
                plt.axis("off")

                n += 1
                if n >= sample:
                    plt.show()
                    return

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_losses = checkpoint["train_losses"]
            self.best_acc = checkpoint.get("best_acc", 0)
        except:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "best_acc": self.best_acc,
        }
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch["img"].to(self.device, non_blocking=True)
        tgt_input = batch["tgt_input"].to(self.device, non_blocking=True)
        tgt_output = batch["tgt_output"].to(self.device, non_blocking=True)
        tgt_padding_mask = batch["tgt_padding_mask"].to(self.device, non_blocking=True)

        batch = {
            "img": img,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "tgt_padding_mask": tgt_padding_mask,
            "filenames": batch["filenames"],
        }

        return batch

    def data_gen(
        self,
        lmdb_path,
        annotation_path,
        masked_language_model=True,
        transform=None,
        is_train=True,
    ):
        dataset = OCRDataset(
            lmdb_path,
            root_dir=self.data_root,
            annotation_path=annotation_path,
            vocab=self.vocab,
            transform=transform,
            image_height=self.config["dataset"]["image_height"],
            image_min_width=self.config["dataset"]["image_min_width"],
            image_max_width=self.config["dataset"]["image_max_width"],
        )

        sampler = ClusterRandomSampler(dataset, self.batch_size, shuffle=is_train)

        collate_fn = Collator(masked_language_model)
        gen = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.config["dataloader"].get("num_workers", 4),
        )

        return gen

    def step(self, batch):
        self.model.train()
        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = (
            batch["img"],
            batch["tgt_input"],
            batch["tgt_output"],
            batch["tgt_padding_mask"],
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(img, tgt_input, tgt_padding_mask)
            outputs = outputs.flatten(0, 1)
            tgt_output = tgt_output.flatten()
            loss = self.criterion(outputs, tgt_output)

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()
