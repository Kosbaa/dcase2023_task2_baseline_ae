import csv
import datetime
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch.nn.functional as F
from sklearn import metrics

from datasets import loader_common as loader_com
from networks.base_model import BaseModel
from networks.autotrash_cnn_dae_kl.network import AutoTrashUNetDenoisingAE
from tools.plot_loss_curve import csv_to_figdata


MODEL_NAME = "autotrash_cnn_dae_kl"


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SupplementalFeatureDataset(Dataset):
    """
    Feature dataset for AutoTrash supplemental recordings.

    The baseline loader only consumes train/test folders. This dataset uses the
    same log-mel feature extractor and appends clean supplemental machine clips
    as additional normal training patches for this model variant only.
    """

    def __init__(self, files, cache_path, args, domain_label="target"):
        self.files = files
        self.cache_path = cache_path
        self.args = args
        self.domain_label = domain_label
        self.condition = np.array([1.0], dtype=np.float32)
        self.cache_hit = os.path.exists(self.cache_path)
        self.data, self.basenames = self._load_or_extract()

    def _load_or_extract(self):
        if os.path.exists(self.cache_path):
            cached = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            return cached["data"], cached["basenames"]

        Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)
        data_parts = []
        basenames = []
        for file_path in self.files:
            vectors = loader_com.file_to_vectors(
                file_path,
                n_mels=self.args.n_mels,
                n_frames=self.args.frames,
                n_fft=self.args.n_fft,
                hop_length=self.args.hop_length,
                power=self.args.power,
                fmax=self.args.fmax,
                fmin=self.args.fmin,
                win_length=self.args.win_length,
                mono=self.args.mono,
            )
            vectors = vectors[:: self.args.frame_hop_length, :]
            if len(vectors) == 0:
                continue
            data_parts.append(vectors.astype(np.float32))
            base = os.path.basename(file_path)
            tagged = f"section_00_{self.domain_label}_supplemental_{base}"
            basenames.extend([tagged] * len(vectors))

        if data_parts:
            data = np.concatenate(data_parts, axis=0)
        else:
            data = np.empty((0, self.args.n_mels * self.args.frames), dtype=np.float32)

        torch.save({"data": data, "basenames": basenames}, self.cache_path)
        return data, basenames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], 0.0, self.condition, self.basenames[index], index


class AutoTrashCnnDaeKl(BaseModel):
    """
    AutoTrash-specific reconstruction model.

    It keeps the official output workflow based on anomaly-score CSVs, but trains
    with denoising augmentation, domain-balanced reconstruction, and weak
    annealed KL regularization.
    """

    def __init__(self, args, train, test):
        self._prepare_autotrash_args(args)
        super().__init__(args=args, train=train, test=test)
        self._append_supplemental_clean_data()
        self.noise_bank = self._load_supplemental_noise_bank()
        if self.noise_bank is not None:
            self.noise_bank = self.noise_bank.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.optim_state_dict:
            self.optimizer.load_state_dict(self.optim_state_dict)
        self._print_startup_status()

    def _prepare_autotrash_args(self, args):
        if args.dataset != "DCASE2025T2AutoTrash":
            raise ValueError(f"{MODEL_NAME} is AutoTrash-specific. Use --dataset DCASE2025T2AutoTrash.")
        if not args.eval:
            raise ValueError(f"{MODEL_NAME} is intended for DCASE2025T2 AutoTrash additional/evaluation data. Use --eval.")
        if args.score != "MSE":
            raise ValueError(f"{MODEL_NAME} supports reconstruction-error scoring only. Use --score MSE.")

        args.epochs = args.autotrash_epochs
        if args.autotrash_batch_size > 0:
            args.batch_size = args.autotrash_batch_size
        if args.autotrash_frames > 0:
            args.frames = args.autotrash_frames
        if args.export_dir in ["", "baseline"]:
            args.export_dir = MODEL_NAME

        if not args.autotrash_cache_directory:
            args.autotrash_cache_directory = str(
                Path(args.dataset_directory) / "dcase2025t2" / "eval_data" / "processed" / "AutoTrash"
            )
        if args.autotrash_rebuild_cache:
            self._reset_feature_cache(args)

    def init_model(self):
        return AutoTrashUNetDenoisingAE(
            frames=self.data.width,
            n_mels=self.data.height,
            latent_dim=self.args.autotrash_latent_dim,
            skip_scale=self.args.autotrash_skip_scale,
        )

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss", "source_recon_loss", "target_recon_loss"],
            ["recon_gap"],
            ["kl_loss"],
            ["lambda_kl"],
            ["epoch_seconds", "avg_batch_seconds"],
        ]
        return (
            "loss,val_loss,recon_loss,source_recon_loss,target_recon_loss,"
            "recon_gap,kl_loss,lambda_kl,"
            "epoch_seconds,avg_batch_seconds"
        )

    def _autotrash_root(self):
        return Path(self.args.dataset_directory) / "dcase2025t2" / "eval_data" / "raw" / "AutoTrash"

    def _reset_feature_cache(self, args):
        cache_root = Path(args.autotrash_cache_directory)
        cache_name = f"section_00_mix_TF{args.frames}-{args.frame_hop_length}_mel{args.n_fft}-{args.hop_length}.pickle"
        targets = [
            cache_root / "train" / f"mels{args.n_mels}_fft{args.n_fft}_hop{args.hop_length}" / cache_name,
            cache_root / "test" / f"mels{args.n_mels}_fft{args.n_fft}_hop{args.hop_length}" / cache_name,
            cache_root / "supplemental",
        ]
        for target in targets:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    def _cache_size_bytes(self, path):
        path = Path(path)
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

    def _print_startup_status(self):
        cache_root = Path(self.args.autotrash_cache_directory)
        mel_dir = f"mels{self.args.n_mels}_fft{self.args.n_fft}_hop{self.args.hop_length}"
        cache_name = f"section_00_mix_TF{self.args.frames}-{self.args.frame_hop_length}_mel{self.args.n_fft}-{self.args.hop_length}.pickle"
        train_cache = cache_root / "train" / mel_dir / cache_name
        test_cache = cache_root / "test" / mel_dir / cache_name
        supplemental_cache = cache_root / "supplemental"
        cache_size = self._cache_size_bytes(cache_root)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("============== AUTOTRASH CNN DAE STARTUP ==============")
        print(f"feature cache directory: {cache_root}")
        print(f"train cache used: {train_cache.exists()} -> {train_cache}")
        print(f"test cache used: {test_cache.exists()} -> {test_cache}")
        print(f"supplemental tensor cache exists: {supplemental_cache.exists()} -> {supplemental_cache}")
        print(f"estimated cache size: {cache_size / (1024 ** 2):.2f} MB")
        print(f"rebuild_cache: {self.args.autotrash_rebuild_cache}")
        print(f"model parameter count: {param_count:,}")
        if torch.cuda.is_available() and self.device.type == "cuda":
            device_idx = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            print(f"gpu memory allocated/reserved: {allocated:.2f} MB / {reserved:.2f} MB")
        print("--- component status ---")
        print(f"denoising:          {self.args.autotrash_use_denoising}")
        print(f"augmentation:       {self.args.autotrash_use_augmentation}")
        print(f"balanced_recon:     {self.args.autotrash_use_balanced_recon}")
        print(f"kl:                 {self.args.autotrash_use_kl}")
        print(f"supplemental_clean: {self.args.autotrash_use_supplemental_clean}")
        print(f"supplemental_noise: {self.args.autotrash_use_supplemental_noise}")
        print(f"decision_threshold_quantile: {self.args.decision_threshold}")

    def _append_supplemental_clean_data(self):
        if not self.args.autotrash_use_supplemental_clean:
            return
        supplemental_dir = self._autotrash_root() / "supplemental"
        files = sorted(glob.glob(str(supplemental_dir / self.args.autotrash_supplemental_clean_glob)))
        files = [f for f in files if "noise" not in os.path.basename(f).lower()]
        if len(files) == 0:
            print(f"warning: no clean supplemental AutoTrash files found in {supplemental_dir}")
            return

        cache_dir = Path(self.args.autotrash_cache_directory) / "supplemental"
        cache_name = (
            f"clean_{self.args.autotrash_supplemental_domain}_TF{self.args.frames}-{self.args.frame_hop_length}_"
            f"mel{self.args.n_fft}-{self.args.hop_length}_mels{self.args.n_mels}.pt"
        )
        supplemental_dataset = SupplementalFeatureDataset(
            files=files,
            cache_path=str(cache_dir / cache_name),
            args=self.args,
            domain_label=self.args.autotrash_supplemental_domain,
        )
        if len(supplemental_dataset) == 0:
            print("warning: supplemental clean files produced no feature patches")
            return

        self.train_loader = DataLoader(
            ConcatDataset([self.train_loader.dataset, supplemental_dataset]),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
        )
        print(
            f"AutoTrash supplemental clean data appended: {len(files)} files, "
            f"{len(supplemental_dataset)} patches as {self.args.autotrash_supplemental_domain} normal "
            f"(cache_hit={supplemental_dataset.cache_hit})"
        )

    def _load_supplemental_noise_bank(self):
        if not self.args.autotrash_use_supplemental_noise:
            return None
        supplemental_dir = self._autotrash_root() / "supplemental"
        files = sorted(glob.glob(str(supplemental_dir / self.args.autotrash_supplemental_noise_glob)))
        if len(files) == 0:
            print("warning: no AutoTrash supplemental noise-only recordings found; noise mixing disabled")
            return None

        cache_dir = Path(self.args.autotrash_cache_directory) / "supplemental"
        cache_path = cache_dir / (
            f"noise_TF{self.args.frames}-{self.args.frame_hop_length}_"
            f"mel{self.args.n_fft}-{self.args.hop_length}_mels{self.args.n_mels}.pt"
        )
        noise_dataset = SupplementalFeatureDataset(
            files=files,
            cache_path=str(cache_path),
            args=self.args,
            domain_label="noise",
        )
        if len(noise_dataset) == 0:
            print("warning: supplemental noise files produced no feature patches; noise mixing disabled")
            return None
        print(f"AutoTrash supplemental noise cache loaded: {len(noise_dataset)} patches (cache_hit={noise_dataset.cache_hit})")
        return torch.from_numpy(noise_dataset.data).float()

    def _to_image(self, x):
        return x.view(-1, self.args.frames, self.args.n_mels).transpose(1, 2).unsqueeze(1)

    def _to_vector(self, x):
        return x.squeeze(1).transpose(1, 2).contiguous().view(-1, self.data.input_dim)

    def _domain_masks(self, basenames, device):
        source = torch.tensor(["source" in name for name in basenames], device=device, dtype=torch.bool)
        target = torch.tensor(["target" in name for name in basenames], device=device, dtype=torch.bool)
        supplemental = torch.tensor(["supplemental" in name for name in basenames], device=device, dtype=torch.bool)
        if self.args.autotrash_supplemental_domain == "source":
            source = source | supplemental
        else:
            target = target | supplemental
        return source, target

    def _augment(self, x, basenames=None):
        """
        Denoising corruption applied only during training.

        The target remains the original log-mel patch. Masks are filled with the
        patch mean instead of zero to avoid destroying too much acoustic structure.
        Gain is implemented as a log-domain dB shift, matching log-mel features.
        """
        img = self._to_image(x).clone()
        original_img = img.clone()
        b, _, mel_bins, frames = img.shape
        is_target = torch.zeros(b, dtype=torch.bool, device=img.device)
        if basenames is not None:
            for i, name in enumerate(basenames):
                if "target" in name and "supplemental" not in name:
                    is_target[i] = True

        if self.args.autotrash_aug_gain_min != 1.0 or self.args.autotrash_aug_gain_max != 1.0:
            gain = torch.empty(b, 1, 1, 1, device=img.device).uniform_(
                self.args.autotrash_aug_gain_min,
                self.args.autotrash_aug_gain_max,
            )
            img = img + 20.0 * torch.log10(torch.clamp(gain, min=1e-6))

        if self.args.autotrash_aug_gaussian_std > 0:
            patch_std = img.flatten(start_dim=1).std(dim=1, keepdim=True).view(b, 1, 1, 1)
            img = img + torch.randn_like(img) * patch_std.clamp_min(1e-6) * self.args.autotrash_aug_gaussian_std

        if self.args.autotrash_aug_time_mask_width > 0 and self.args.autotrash_aug_time_mask_prob > 0:
            apply_mask = (torch.rand(b, device=img.device) <= self.args.autotrash_aug_time_mask_prob).detach().cpu().numpy()
            widths = torch.randint(
                1,
                self.args.autotrash_aug_time_mask_width + 1,
                (b,),
                device=img.device,
            ).detach().cpu().numpy()
            starts = torch.randint(0, frames, (b,), device=img.device).detach().cpu().numpy()
            patch_means = img.mean(dim=(1, 2, 3), keepdim=True)
            for i in range(b):
                if not apply_mask[i]:
                    continue
                width = int(widths[i])
                width = min(width, frames)
                start = min(int(starts[i]), frames - width)
                img[i, :, :, start:start + width] = patch_means[i]

        if self.args.autotrash_aug_freq_mask_width > 0 and self.args.autotrash_aug_freq_mask_prob > 0:
            apply_mask = (torch.rand(b, device=img.device) <= self.args.autotrash_aug_freq_mask_prob).detach().cpu().numpy()
            widths = torch.randint(
                1,
                self.args.autotrash_aug_freq_mask_width + 1,
                (b,),
                device=img.device,
            ).detach().cpu().numpy()
            starts = torch.randint(0, mel_bins, (b,), device=img.device).detach().cpu().numpy()
            patch_means = img.mean(dim=(1, 2, 3), keepdim=True)
            for i in range(b):
                if not apply_mask[i]:
                    continue
                width = int(widths[i])
                width = min(width, mel_bins)
                start = min(int(starts[i]), mel_bins - width)
                img[i, :, start:start + width, :] = patch_means[i]

        if self.noise_bank is not None and self.args.autotrash_noise_mix_prob > 0:
            if torch.rand((), device=img.device) <= self.args.autotrash_noise_mix_prob:
                idx = torch.randint(0, len(self.noise_bank), (b,), device=img.device)
                noise = self._to_image(self.noise_bank[idx])
                x_rms = img.flatten(start_dim=1).pow(2).mean(dim=1, keepdim=True).sqrt().view(b, 1, 1, 1)
                n_rms = noise.flatten(start_dim=1).pow(2).mean(dim=1, keepdim=True).sqrt().view(b, 1, 1, 1)
                snr = torch.empty(b, 1, 1, 1, device=img.device).uniform_(
                    self.args.autotrash_noise_snr_min,
                    self.args.autotrash_noise_snr_max,
                )
                scale = x_rms / (n_rms.clamp_min(1e-6) * torch.pow(10.0, snr / 20.0))
                img = img + noise * scale

        if is_target.any():
            img[is_target] = original_img[is_target]

        return self._to_vector(img)

    def _sample_scores(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none").mean(dim=1)

    def _domain_balanced_recon_loss(self, sample_scores, source_mask, target_mask):
        source_loss = sample_scores[source_mask].mean() if source_mask.any() else None
        target_loss = sample_scores[target_mask].mean() if target_mask.any() else None

        if not self.args.autotrash_use_balanced_recon:
            recon_loss = sample_scores.mean()
        elif source_loss is not None and target_loss is not None:
            recon_loss = 0.5 * source_loss + 0.5 * target_loss
        elif source_loss is not None:
            recon_loss = source_loss
        elif target_loss is not None:
            recon_loss = target_loss
        else:
            recon_loss = sample_scores.mean()

        zero = sample_scores.new_tensor(0.0)
        source_log = source_loss if source_loss is not None else zero
        target_log = target_loss if target_loss is not None else zero
        gap = torch.abs(source_log - target_log) if source_loss is not None and target_loss is not None else zero
        return recon_loss, source_log, target_log, gap

    def _kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def _lambda_kl(self, epoch):
        if not self.args.autotrash_use_kl:
            return 0.0
        if epoch < self.args.autotrash_kl_start_epoch:
            return 0.0
        anneal_epochs = max(1, self.args.epochs - self.args.autotrash_kl_start_epoch)
        progress = (epoch - self.args.autotrash_kl_start_epoch) / anneal_epochs
        return self.args.autotrash_kl_max_weight * min(1.0, max(0.0, progress))

    def _progress_line(self, epoch, batch_idx, n_batches, train_start, epoch_start, losses):
        elapsed = time.time() - train_start
        completed_batches = (epoch - 1) * n_batches + batch_idx
        total_batches = self.args.epochs * n_batches
        avg_batch = elapsed / max(1, completed_batches)
        remaining = avg_batch * max(0, total_batches - completed_batches)
        finish = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        return (
            f"Epoch [{epoch}/{self.args.epochs}] Batch [{batch_idx}/{n_batches}] | "
            f"Loss: {losses['loss']:.4f} | Recon: {losses['recon']:.4f} | "
            f"KL: {losses['kl']:.4f} | "
            f"Elapsed: {_format_seconds(elapsed)} | ETA: {_format_seconds(remaining)} | "
            f"Finish: {finish.strftime('%H:%M:%S')}"
        )

    def train(self, epoch):
        if epoch <= self.epoch or epoch > self.args.epochs:
            return

        self.model.train()
        train_start = getattr(self, "_train_start_time", None)
        if train_start is None:
            train_start = time.time()
            self._train_start_time = train_start
        epoch_start = time.time()

        sums = {
            "loss": 0.0,
            "recon": 0.0,
            "source": 0.0,
            "target": 0.0,
            "gap": 0.0,
            "kl": 0.0,
        }
        lambda_kl = self._lambda_kl(epoch)
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue
            basenames = batch[3]
            source_mask, target_mask = self._domain_masks(basenames, data.device)

            self.optimizer.zero_grad()
            model_input = self._augment(data, basenames) if self.args.autotrash_use_augmentation else data
            # Denoising ON reconstructs the original clean patch from corrupted input.
            # Denoising OFF keeps augmentation active but trains a standard AE target.
            recon_target = data if self.args.autotrash_use_denoising else model_input.detach()
            recon_batch, _, mu, logvar = self.model(model_input)
            sample_scores = self._sample_scores(recon_batch, recon_target)
            recon_loss, source_loss, target_loss, gap = self._domain_balanced_recon_loss(
                sample_scores,
                source_mask,
                target_mask,
            )
            kl_loss = self._kl_loss(mu, logvar)
            self.loss = recon_loss + lambda_kl * kl_loss
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.autotrash_grad_clip)
            self.optimizer.step()

            sums["loss"] += float(self.loss.detach())
            sums["recon"] += float(recon_loss.detach())
            sums["source"] += float(source_loss.detach())
            sums["target"] += float(target_loss.detach())
            sums["gap"] += float(gap.detach())
            sums["kl"] += float(kl_loss.detach())

            if self.args.autotrash_show_progress and (
                batch_idx == 1
                or batch_idx == n_batches
                or batch_idx % self.args.autotrash_eta_update_interval == 0
            ):
                print(self._progress_line(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                    train_start=train_start,
                    epoch_start=epoch_start,
                    losses={
                        "loss": sums["loss"] / batch_idx,
                        "recon": sums["recon"] / batch_idx,
                        "kl": sums["kl"] / batch_idx,
                    },
                ))

        val_loss = self._validate()
        denom = max(1, n_batches)
        means = {k: v / denom for k, v in sums.items()}
        epoch_seconds = time.time() - epoch_start
        avg_batch_seconds = epoch_seconds / denom
        remaining_epochs = max(0, self.args.epochs - epoch)
        print(
            f"====> Epoch: {epoch} Average loss: {means['loss']:.4f} "
            f"Validation loss: {val_loss:.4f}"
        )
        print(
            f"epoch duration: {_format_seconds(epoch_seconds)} | avg batch: {avg_batch_seconds:.3f}s | "
            f"remaining training estimate: {_format_seconds(avg_batch_seconds * denom * remaining_epochs)}"
        )

        with open(self.log_path, "a") as log:
            np.savetxt(log, ["{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}".format(
                means["loss"],
                val_loss,
                means["recon"],
                means["source"],
                means["target"],
                means["gap"],
                means["kl"],
                lambda_kl,
                epoch_seconds,
                avg_batch_seconds,
            )], fmt="%s")

        csv_to_figdata(
            file_path=self.log_path,
            column_heading_list=self.column_heading_list,
            ylabel="loss",
            fig_count=len(self.column_heading_list),
            cut_first_epoch=True,
        )
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
        }, self.checkpoint_path)
        if epoch == self.args.epochs:
            self._run_final_threshold_calibration()

    def _validate(self):
        val_loss = 0.0
        with torch.no_grad():
            self.model.eval()
            for batch in self.valid_loader:
                data = batch[0].to(self.device).float()
                recon_batch, _, _, _ = self.model(data)
                sample_scores = self._sample_scores(recon_batch, data)
                loss = sample_scores.mean()
                val_loss += float(loss)
        return val_loss / max(1, len(self.valid_loader))

    def _run_final_threshold_calibration(self):
        """
        Calibrate the decision threshold once after training completes.
        Uses clip-level scores (one mean score per file) from held-out
        validation source clips only.

        Clip-level calibration is essential: test() scores each clip as
        the mean of its patch errors. Calibrating on patch-level scores
        produces a threshold far too high for clip-level scores because
        patch variance >> clip variance (Central Limit Theorem).

        Source-only rationale:
        - 990 source normals give a stable score distribution
        - Target scores are systematically offset from source scores
        - Mixing them biases the threshold upward, causing target recall collapse
        """
        score_sum = {}
        score_count = {}
        with torch.no_grad():
            self.model.eval()
            for batch in self.valid_loader:
                data = batch[0].to(self.device).float()
                basenames = batch[3]
                recon_batch, _, _, _ = self.model(data)
                patch_errors = self._sample_scores(recon_batch, data).detach().cpu().numpy()
                for basename, score in zip(basenames, patch_errors):
                    if "target" in basename:
                        continue
                    score_sum[basename] = score_sum.get(basename, 0.0) + float(score)
                    score_count[basename] = score_count.get(basename, 0) + 1

        if len(score_sum) == 0:
            print("warning: no source clips found in valid_loader; falling back to all validation clips")
            with torch.no_grad():
                self.model.eval()
                for batch in self.valid_loader:
                    data = batch[0].to(self.device).float()
                    basenames = batch[3]
                    recon_batch, _, _, _ = self.model(data)
                    patch_errors = self._sample_scores(recon_batch, data).detach().cpu().numpy()
                    for basename, score in zip(basenames, patch_errors):
                        score_sum[basename] = score_sum.get(basename, 0.0) + float(score)
                        score_count[basename] = score_count.get(basename, 0) + 1

        clip_scores = np.array([
            score_sum[name] / score_count[name]
            for name in sorted(score_sum)
        ])

        print(
            f"threshold calibration -> {len(clip_scores)} source clips (clip-level) | "
            f"mean={np.mean(clip_scores):.6f} | std={np.std(clip_scores):.6f} | "
            f"min={np.min(clip_scores):.6f} | max={np.max(clip_scores):.6f}"
        )
        self.fit_anomaly_score_distribution(y_pred=clip_scores)

        fitted_threshold = self.calc_decision_threshold()
        p99 = float(np.percentile(clip_scores, 99))
        print(f"fitted threshold -> {fitted_threshold:.6f} | p99 of clip scores -> {p99:.6f}")
        if fitted_threshold > p99 * 2.0:
            print(
                f"WARNING: fitted threshold ({fitted_threshold:.6f}) far exceeds 2x p99 "
                f"({p99 * 2.0:.6f}). Gamma fit may have failed on this seed."
            )

    def load_state_dict(self, checkpoint):
        super().load_state_dict(checkpoint=checkpoint)

    def test(self):
        mode = self.data.mode
        csv_lines = []

        print("============== MODEL LOAD ==============")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found -> {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self._run_final_threshold_calibration()
        decision_threshold = self.calc_decision_threshold()
        print(f"decision threshold -> {decision_threshold:.6f}")

        result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
        dir_name = "test"
        performance = []

        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            anomaly_score_csv = result_dir / (
                f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}.csv"
            )
            decision_result_csv = result_dir / (
                f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}.csv"
            )

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            anomaly_score_list = []
            decision_result_list = []
            y_pred = []
            y_true = []
            domain_list = [] if mode else None

            with torch.no_grad():
                for batch in test_loader_tmp:
                    data = batch[0].to(self.device).float()
                    basename = batch[3][0]
                    recon_data, _, _, _ = self.model(data)
                    score = self._sample_scores(recon_data, data).mean().item()
                    y_pred.append(score)
                    y_true.append(batch[1][0].item())
                    anomaly_score_list.append([basename, score])
                    decision_result_list.append([basename, 1 if score > decision_threshold else 0])
                    if mode:
                        domain_list.append("target" if "target" in basename else "source")

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print(f"anomaly score result ->  {anomaly_score_csv}")
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print(f"decision result ->  {decision_result_csv}")

            if mode:
                section_metrics = self._append_local_metrics(
                    csv_lines=csv_lines,
                    section_name=section_name,
                    y_true=y_true,
                    y_pred=y_pred,
                    domain_list=domain_list,
                )
                performance.append(section_metrics)

            print("\n============ END OF TEST FOR A SECTION ============")

        if mode and performance:
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
                axis=0,
            )
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            result_path = result_dir / (
                f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}_roc.csv"
            )
            print(f"results -> {result_path}")
            save_csv(save_file_path=result_path, save_data=csv_lines)

    def _append_local_metrics(self, csv_lines, section_name, y_true, y_pred, domain_list):
        y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
        y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
        y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
        y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
        y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
        y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
        y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]
        y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]

        threshold = self.calc_decision_threshold()
        auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
        p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
        tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > threshold else 0 for x in y_pred_s]).ravel()
        prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
        recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
        f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

        if len(y_true_t) > 0:
            auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
            p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
            tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > threshold else 0 for x in y_pred_t]).ravel()
            prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
            recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
            f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
            if len(csv_lines) == 0:
                csv_lines.append(self.result_column_dict["source_target"])
            row = [auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t]
            csv_lines.append([section_name.split("_", 1)[1]] + row)
            return row

        if len(csv_lines) == 0:
            csv_lines.append(self.result_column_dict["single_domain"])
        row = [auc_s, p_auc, prec_s, recall_s, f1_s]
        csv_lines.append([section_name.split("_", 1)[1]] + row)
        return row
