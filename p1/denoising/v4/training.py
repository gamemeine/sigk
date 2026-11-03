"""SIGK V4: four controlled trainings, AdamW + AMP + normalized MSE + SSIM.
No test data is used for model selection. Resumable at epoch boundaries.
"""
import copy, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from .data import PairedTrainingDataset, DIV2KDenoising, fixed_noise, SIGMAS
from .models import make_model


def optimizer_groups(model):
    decayed, not_decayed = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or n.endswith('.bias'):
            not_decayed.append(p)
        else:
            decayed.append(p)
    return [{'params': decayed, 'weight_decay':1e-4},
            {'params':not_decayed, 'weight_decay':0.0}]


def update_ema(ema, model, decay):
    with torch.no_grad():
        src = model.state_dict()
        for key, value in ema.state_dict().items():
            if torch.is_floating_point(value):
                value.lerp_(src[key].detach(), 1.0-decay)
            else:
                value.copy_(src[key])


def build_amp(device, enabled):
    on = bool(enabled and device.type == 'cuda')
    return on, torch.amp.GradScaler('cuda', enabled=on)


def loss_fn(denoised, clean, sigma, ssim_weight):
    # Perform loss computations in fp32, particularly for very small noise (sigma=0.01).
    pred = denoised.float()
    clean = clean.float()
    sig = sigma.float()[:, None, None, None]
    mse = (((pred-clean) / sig) ** 2).mean()
    if ssim_weight:
        from torchmetrics.functional.image import structural_similarity_index_measure
        val = structural_similarity_index_measure(pred.clamp(0,1), clean, data_range=1.0)
        return mse + ssim_weight*(1.0-val)
    return mse


@torch.no_grad()
def validate(model, root, device, patch_size, seed=1_000_042, allow_small=False):
    model.eval()
    ds = DIV2KDenoising(root, 'val', size=patch_size, augment='none')
    if not allow_small and len(ds) != 50:
        raise ValueError(f'Expected 50 DIV2K val images, got {len(ds)}')
    # Prevent a new model seeing test samples during selection.
    out = {}
    for sigma in SIGMAS:
        psnr, ssim = [], []
        for i in range(len(ds)):
            clean = ds[i][1]
            noisy = fixed_noise(clean, sigma, seed, i)
            with torch.no_grad():
                y = model(noisy[None].to(device), sigma).clamp(0,1)[0].float().cpu()
            ref = clean.permute(1,2,0).numpy()
            got = y.permute(1,2,0).numpy()
            psnr.append(peak_signal_noise_ratio(ref, got, data_range=1.0))
            ssim.append(structural_similarity(ref, got, channel_axis=-1, data_range=1.0))
        out[f'psnr_{sigma:.2f}'] = float(np.mean(psnr))
        out[f'ssim_{sigma:.2f}'] = float(np.mean(ssim))
    out['psnr'] = float(np.mean([out[f'psnr_{s:.2f}'] for s in SIGMAS]))
    out['ssim'] = float(np.mean([out[f'ssim_{s:.2f}'] for s in SIGMAS]))
    out['selection_score'] = out['psnr'] + 10.0*out['ssim']
    return out


def run_train(name, root, output, settings, augmentation_x4=False, smoke=False):
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = bool(device.type == 'cuda')
    seed = int(settings.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    arch = 'drunet_lite' if name.startswith('drunet_lite') else 'rescross'
    model_cfg = {'width':int(settings.get('drunet_width',32) if arch=='drunet_lite' else settings.get('rescross_width',64))}
    model = make_model(arch, model_cfg).to(device)
    ema = copy.deepcopy(model).eval()
    multi = bool(settings.get('multi_gpu',False) and device.type == 'cuda' and torch.cuda.device_count()>1)
    forward_model = torch.nn.DataParallel(model) if multi else model
    lr = float(settings.get('lr',2e-4))
    epochs = int(settings.get('epochs',14))
    per_epoch = int(settings.get('samples_per_epoch',4800))
    batch = int(settings.get('batch',32))
    patch_size = int(settings.get('patch_size',256))
    workers = int(settings.get('workers',4))
    ssim_weight = float(settings.get('ssim_weight',1.0))
    ema_decay = float(settings.get('ema_decay',0.999))
    use_amp, scaler = build_amp(device, settings.get('amp',True))
    opt = torch.optim.AdamW(optimizer_groups(model), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1), eta_min=float(settings.get('lr_min',1e-6)))
    ds = PairedTrainingDataset(root, size=patch_size, samples_per_epoch=per_epoch, seed=seed, x4=augmentation_x4)
    last = out/'last.pt'; best = out/'best.pt'; history = out/'history.csv'
    start, best_score = 1, -1e30
    if last.exists():
        cp = torch.load(last,map_location='cpu',weights_only=True)
        if cp['architecture'] != arch or cp['model_config'] != model_cfg or bool(cp['aug_x4']) != augmentation_x4:
            raise RuntimeError('Existing last.pt has incompatible model or augmentation settings')
        # Enforce consistency for reproducible LR and total number of training steps.
        for attr, target in [('epochs',epochs),('samples_per_epoch',per_epoch),('batch',batch),('patch_size',patch_size)]:
            if int(cp['settings'].get(attr,target)) != target:
                raise RuntimeError(f'Cannot resume with changed {attr}; create a new RUN_NAME')
        model.load_state_dict(cp['state_dict']); ema.load_state_dict(cp['ema_state_dict'])
        opt.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        if cp.get('scaler') and use_amp: scaler.load_state_dict(cp['scaler'])
        best_score = float(cp['best_score'])
        start = int(cp['epoch'])+1
        print(f'RESUME {name}: epoch={start} using {last}',flush=True)
    (out/'settings.json').write_text(json.dumps({'architecture':arch,'model_config':model_cfg,'augmentation_x4':augmentation_x4,
                     'settings':settings,'model_params':sum(p.numel() for p in model.parameters())},indent=2),encoding='utf-8')
    for ep in range(start,epochs+1):
        ds.set_epoch(ep)
        loader = DataLoader(ds,batch_size=batch,shuffle=True,
            num_workers=workers,pin_memory=device.type=='cuda',
            persistent_workers=bool(workers>0),drop_last=True,
            generator=torch.Generator().manual_seed(seed+ep))
        model.train()
        nimages, tot_loss, used_micro, st = 0,0.,batch,time.perf_counter()
        for noisy, clean, sigma in loader:
            noisy = noisy.to(device,non_blocking=True); clean=clean.to(device,non_blocking=True)
            sigma = sigma.to(device,non_blocking=True)
            # Actual effective/global batch == batch. If OOM, lower microbatch
            # automatically but ACCUMULATE gradients across same global batch.
            micro = min(used_micro, len(noisy))
            while True:
                opt.zero_grad(set_to_none=True)
                try:
                    loss_value = 0.
                    for begin in range(0,len(noisy),micro):
                        n = noisy[begin:begin+micro]
                        c = clean[begin:begin+micro]
                        s = sigma[begin:begin+micro]
                        with torch.amp.autocast(device_type='cuda',enabled=use_amp):
                            result = forward_model(n,s)
                        loss = loss_fn(result,c,s,ssim_weight)
                        fraction = len(n)/len(noisy)
                        scaler.scale(loss*fraction).backward()
                        loss_value += float(loss.detach())*fraction
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(opt)
                    scaler.update()
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    if device.type!='cuda' or micro<=1:
                        raise
                    micro=max(1,micro//2)
                    used_micro=micro
                    torch.cuda.empty_cache()
                    print(f'GPU OOM: lower physical microbatch to {micro}, keep effective batch={batch}',flush=True)
            nimages += len(noisy)
            tot_loss += loss_value*len(noisy)
            update_ema(ema,model,ema_decay)
        scheduler.step()
        if device.type == 'cuda':torch.cuda.synchronize()
        secs=time.perf_counter()-st
        raw=validate(model,root,device,patch_size,allow_small=smoke)
        ema_result=validate(ema,root,device,patch_size,allow_small=smoke)
        chosen = 'raw' if raw['selection_score'] >= ema_result['selection_score'] else 'ema'
        selected = raw if chosen=='raw' else ema_result
        score = selected['selection_score']
        selected_state = model.state_dict() if chosen=='raw' else ema.state_dict()
        checkpoint = dict(epoch=ep,architecture=arch,model_config=model_cfg,aug_x4=augmentation_x4,
              state_dict=copy.deepcopy(model.state_dict()),ema_state_dict=copy.deepcopy(ema.state_dict()),
              optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
              settings=settings,best_score=max(best_score,score),eval_weights=chosen,
              metrics_raw=raw,metrics_ema=ema_result)
        torch.save(checkpoint,last)
        improved=score > best_score
        if improved:
            best_score=score
            checkpoint['eval_state_dict'] = copy.deepcopy(selected_state)
            torch.save(checkpoint,best)
        row=dict(epoch=ep,lr=opt.param_groups[0]['lr'],loss=tot_loss/max(nimages,1),
              train_images=nimages,train_seconds=secs,images_per_second=nimages/max(secs,1e-8),
              microbatch=used_micro,raw_psnr=raw['psnr'],raw_ssim=raw['ssim'],
              ema_psnr=ema_result['psnr'],ema_ssim=ema_result['ssim'],
              chosen=chosen,selected_psnr=selected['psnr'],selected_ssim=selected['ssim'],
              psnr_001=selected['psnr_0.01'],psnr_003=selected['psnr_0.03'],
              ssim_001=selected['ssim_0.01'],ssim_003=selected['ssim_0.03'])
        with history.open('a',newline='') as f:
            dw=csv.DictWriter(f,fieldnames=row.keys())
            if not history.stat().st_size:dw.writeheader()
            dw.writerow(row)
        print(f'[{name}] {ep:02d}/{epochs}  LR={row["lr"]:.2e}  PSNR={row["selected_psnr"]:.3f} '
              f'SSIM={row["selected_ssim"]:.4f}  {chosen}  {row["images_per_second"]:.1f} img/s '
              f'(micro={used_micro}/{batch}) '+('*' if improved else ''),flush=True)
    return best
