"""SIGK V3: DnCNN V2 -> AdamW + SSIM fine-tuning + EMA, val-only model selection."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DIV2KDenoising, SIGMAS, fixed_noise
from .model import DnCNN
from .train import resolve_device, psnr_per_image


def get_ssim():
    from torchmetrics.functional.image import structural_similarity_index_measure
    return structural_similarity_index_measure


def adamw_parameters(model: nn.Module):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Wylaczamy weight decay dla biasow i normalizacji (tensor 1D).
        (no_decay if param.ndim < 2 or name.endswith('.bias') else decay).append(param)
    return [{'params': decay, 'weight_decay': 1e-4},
            {'params': no_decay, 'weight_decay': 0.0}]


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.lerp_(param.detach(), 1 - decay)
    # BN running_mean/running_var sa estymowane podczas trenowania online.
    # Kopiujemy je z modelu, nie usredniamy (w szczegolnosci int num_batches_tracked).
    for dst, src in zip(ema.buffers(), model.buffers()):
        dst.copy_(src)


@torch.no_grad()
def validate_v3(model: DnCNN, ds: DIV2KDenoising, device: torch.device, seed: int) -> dict:
    from skimage.metrics import structural_similarity
    model.eval()
    metrics = {sigma: {'psnr': [], 'ssim': []} for sigma in SIGMAS}
    for i in range(len(ds)):
        _, clean, _ = ds[i]
        tgt = clean.unsqueeze(0).to(device)
        for sigma in SIGMAS:
            inp = fixed_noise(clean, sigma, seed, i).unsqueeze(0).to(device)
            pred = model(inp, sigma).float().clamp(0, 1)
            metrics[sigma]['psnr'].append(float(psnr_per_image(pred, tgt).item()))
            # Te same parametry SSIM co ewaluacja SIGK (skimage): zwykly win_size=7.
            arr = pred[0].permute(1, 2, 0).cpu().numpy()
            ref = clean.permute(1, 2, 0).numpy()
            metrics[sigma]['ssim'].append(float(structural_similarity(ref, arr, channel_axis=-1, data_range=1.0)))
    results = {}
    for sigma in SIGMAS:
        for key in ('psnr', 'ssim'):
            results[f'val_{key}_sigma_{sigma:.2f}'] = float(np.mean(metrics[sigma][key]))
    results['val_psnr_mean'] = float(np.mean([results[f'val_psnr_sigma_{s:.2f}'] for s in SIGMAS]))
    results['val_ssim_mean'] = float(np.mean([results[f'val_ssim_sigma_{s:.2f}'] for s in SIGMAS]))
    return results


def train(args):
    if args.batch_size != 16 and not args.allow_incomplete:
        print('UWAGA: batch size ustawiony na', args.batch_size, '(domyslnie V3=16)')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    train_ds = DIV2KDenoising(args.data_dir, 'train', size=args.patch_size, augment=args.augment,
                             samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    val_ds = DIV2KDenoising(args.data_dir, 'val', size=args.patch_size, augment='none', seed=args.seed)
    if not args.allow_incomplete and (len(train_ds.paths) != 800 or len(val_ds.paths) != 50):
        raise RuntimeError('Wymagane pelne DIV2K train=800, val=50; do testow allow_incomplete=True')

    model = DnCNN(depth=args.depth, features=args.features, conditioned=True).to(device)
    start_from = Path(args.init_checkpoint) if args.init_checkpoint else None
    if start_from is None and not args.resume:
        raise ValueError('Fine-tuning V3 wymaga init_checkpoint V2. Mozna uzyc --resume V3.')
    initial = None
    if start_from and not args.resume:
        initial = torch.load(start_from, map_location='cpu', weights_only=True)
        if initial['model_config'] != {'depth': args.depth, 'features': args.features, 'conditioned': True}:
            raise ValueError(f'Inna architektura niz V2: {initial["model_config"]}')
        model.load_state_dict(initial['state_dict'])
        print(f'Zaladowano wagi V2, epoka {initial["epoch"]}, plik: {start_from}')
    ema = copy.deepcopy(model).to(device)
    ema.requires_grad_(False)
    net = model
    if args.multi_gpu and device.type == 'cuda' and torch.cuda.device_count() >= 2:
        net = nn.DataParallel(model)
        print(f'DataParallel na {torch.cuda.device_count()} GPU (bez gwarancji przyspieszenia).')
    else:
        print('Trening na', device)
    opt = torch.optim.AdamW(adamw_parameters(model), lr=args.lr,
                            betas=(0.9, 0.999), eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ssim = get_ssim()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / 'history.csv'
    start_epoch = 0
    initial_metrics = None
    if args.resume:
        prev = torch.load(args.resume, map_location='cpu', weights_only=True)
        if prev.get('model_config') != {'depth': args.depth, 'features': args.features, 'conditioned': True}:
            raise ValueError('Niezgodny checkpoint resume.')
        model.load_state_dict(prev['state_dict'])
        ema.load_state_dict(prev['ema_state_dict'])
        opt.load_state_dict(prev['optimizer'])
        sch.load_state_dict(prev['scheduler'])
        if prev.get('scaler') and use_amp:
            scaler.load_state_dict(prev['scaler'])
        start_epoch = int(prev['epoch'])
        initial_metrics = prev.get('initial_metrics')
        print(f'Resume V3 z epoki {start_epoch}')
    elif history_path.exists():
        raise FileExistsError(f'Wyniki juz istnieja: {history_path}. Zmien RUN_NAME lub wznow z last.pt.')

    if initial_metrics is None:
        # V2 na walidacji jako guard; wybor wag NIE uwzglednia testu 0851-0900.
        initial_metrics = validate_v3(model, val_ds, device, args.seed + 1_000_000)
        print('Wynik poczatkowy V2 na VAL:', initial_metrics)
    ref_psnr = initial_metrics['val_psnr_mean']
    # Zapisujemy V2 jako kandydata, aby nie zgubic wyjsciowej jakosci.
    if not args.resume:
        baseline_state = dict(epoch=0, epoch_v2=initial['epoch'] if initial else None,
            model_config={'depth': args.depth, 'features': args.features, 'conditioned': True},
            state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
            eval_state_dict=model.state_dict(), eval_weights='raw',
            optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict(),
            best_metrics=initial_metrics, initial_metrics=initial_metrics,
            train_config={**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                          'batch_size': args.batch_size, 'optimizer_name': 'AdamW'},
            best_score=initial_metrics['val_ssim_mean'])
        torch.save(baseline_state, out / 'best.pt')
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    best_ssim = float(best['best_score'])
    train_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    train_config['optimizer_name'] = 'AdamW'
    (out / 'config.json').write_text(json.dumps({'args': train_config,
       'initial_val': initial_metrics, 'n_train_images': len(train_ds.paths),
       'n_val_images': len(val_ds.paths),
       'selection': f'mean SSIM on VAL, PSNR drop <= {args.psnr_guard_db} dB vs V2 VAL'},
       indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['epoch', 'train_mse', 'train_norm_mse', 'train_ssim_loss', 'lr',
              'raw_psnr', 'raw_ssim', 'ema_psnr', 'ema_ssim', 'best_eval', 'best_val_psnr', 'best_val_ssim']
    with history_path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            train_ds.set_epoch(epoch)
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=device.type == 'cuda',
                                persistent_workers=args.workers > 0,
                                generator=torch.Generator().manual_seed(args.seed+epoch))
            net.train()
            seen, mse_sum, nmse_sum, ssim_sum = 0, 0., 0., 0.
            for noisy, clean, sigma in tqdm(loader, desc=f'V3 {epoch+1}/{args.epochs}', leave=False):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                sigma = sigma.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    restored = net(noisy, sigma)
                # Malutkie wartosci sigma: loss obliczamy w FP32.
                restored = restored.float()
                nmse = (((restored-clean.float()) / sigma.float().view(-1,1,1,1))**2).mean()
                ssim_loss = 1.0 - ssim(restored.clamp(0,1), clean.float(), data_range=1.0)
                loss = nmse + args.ssim_weight * ssim_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss jest NaN/Inf. Wylacz AMP lub zmniejsz LR.')
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                ema_update(ema, model, args.ema_decay)
                n=len(noisy)
                seen+=n
                mse_sum += nn.functional.mse_loss(restored.detach(), clean).item()*n
                nmse_sum += nmse.detach().item()*n
                ssim_sum += ssim_loss.detach().item()*n
            raw_metrics = validate_v3(model, val_ds, device, args.seed+1_000_000)
            ema_metrics = validate_v3(ema, val_ds, device, args.seed+1_000_000)
            selected = None
            for candidate_name, metrics in [('raw', raw_metrics), ('ema', ema_metrics)]:
                if (metrics['val_psnr_mean'] >= ref_psnr - args.psnr_guard_db and
                        metrics['val_ssim_mean'] > best_ssim):
                    best_ssim = metrics['val_ssim_mean']
                    selected = (candidate_name, metrics)
            current_lr=opt.param_groups[0]['lr']
            sch.step()
            snapshot=dict(epoch=epoch+1,
                model_config={'depth': args.depth, 'features': args.features, 'conditioned': True},
                state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict(),
                train_config=train_config, initial_metrics=initial_metrics,
                best_score=best_ssim, best_metrics=selected[1] if selected else best.get('best_metrics', initial_metrics),
                eval_weights=selected[0] if selected else None,
            )
            torch.save(snapshot, out / 'last.pt')
            if selected:
                # eval_state_dict to wagi uzywane do testu. Pozostaly state_dict sluzy do wznowienia.
                snapshot['eval_state_dict'] = selected[1] and copy.deepcopy(
                    model.state_dict() if selected[0]=='raw' else ema.state_dict())
                torch.save(snapshot, out / 'best.pt')
                best = snapshot
            row={'epoch':epoch+1, 'train_mse':mse_sum/seen,
                 'train_norm_mse':nmse_sum/seen, 'train_ssim_loss':ssim_sum/seen,
                 'lr':current_lr, 'raw_psnr':raw_metrics['val_psnr_mean'],
                 'raw_ssim':raw_metrics['val_ssim_mean'],
                 'ema_psnr':ema_metrics['val_psnr_mean'],
                 'ema_ssim':ema_metrics['val_ssim_mean'],
                 'best_eval':selected[0] if selected else '',
                 'best_val_psnr':best['best_metrics']['val_psnr_mean'],
                 'best_val_ssim':best_ssim}
            writer.writerow(row)
            handle.flush()
            print(f'Epoka {epoch+1:>2} | LR {current_lr:.2e} | '
                  f'raw {row["raw_psnr"]:.3f}/{row["raw_ssim"]:.4f} | '
                  f'EMA {row["ema_psnr"]:.3f}/{row["ema_ssim"]:.4f} | '
                  f'best {row["best_val_psnr"]:.3f}/{row["best_val_ssim"]:.4f} '
                  f'{"*" if selected else ""}')
    return out / 'best.pt'
