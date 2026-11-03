"""Uniform DIV2K held-out benchmark: same clean crop, same fixed Gaussian noise, same metrics.
Methods: input, skimage bilateral, V2, V3 and four newly trained V4 variants.
"""
from __future__ import annotations
import csv, json, time, io, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral
from .data import DIV2KDenoising, fixed_noise, SIGMAS
from .models import make_model


def safe_torch_load(p):
    return torch.load(p,map_location='cpu',weights_only=True)


def find_legacy(input_dir, output_dir, label, explicit=None):
    if explicit:
        p=Path(explicit)
        if not p.is_file():raise FileNotFoundError(f'{label} checkpoint path missing: {p}')
        return p
    root=Path(input_dir)
    if not root.exists():return None
    # Checkpoint files inside Kaggle Add Input are normally already unzipped.
    candidates=[]
    for p in root.rglob('*.pt'):
        if p.name!='best.pt' and label not in p.name.lower():continue
        key=str(p).lower()
        if label in key:candidates.append(p)
    if candidates:
        return sorted(candidates,key=lambda x:len(str(x)))[0]
    for archive_path in root.rglob('*.zip'):
        if not any(w in archive_path.name.lower() for w in ('sigk','dncnn','checkpoint','result')):continue
        with zipfile.ZipFile(archive_path) as z:
            members=[i for i in z.infolist()
                     if i.filename.endswith('best.pt') and (label in i.filename.lower() or label in archive_path.name.lower())
                     and i.file_size < 250_000_000]
            if not members:continue
            member=members[0]
            dest=Path(output_dir)/f'imported_{label}_best.pt'
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member,'r') as src, dest.open('wb') as out:
                import shutil
                shutil.copyfileobj(src,out)
            return dest
    return None


def find_bilateral(input_dir, work_dir):
    for root in [Path(work_dir),Path(input_dir)]:
        if not root.exists():continue
        results=list(root.rglob('bilateral_val_tuned.json'))
        if results:return results[0]
    for p in Path(input_dir).rglob('*.zip') if Path(input_dir).exists() else []:
        with zipfile.ZipFile(p) as z:
            member=next((i for i in z.infolist() if i.filename.endswith('bilateral_val_tuned.json') and i.file_size<1_000_000),None)
            if member:
                dest=Path(work_dir)/'bilateral_val_tuned.json'
                dest.write_bytes(z.read(member))
                return dest
    return None


def tune_bilateral(data_root, file_path, seed=123, smoke=False):
    ds=DIV2KDenoising(data_root,'val',size=32 if smoke else 256,augment='none')
    if not smoke and len(ds)!=50:raise ValueError('Expected complete 50-image validation')
    candidates=[(a,b) for a in (1.,2.,3.,4.) for b in (.25,.35,.5,.75,1.)]
    best={}
    # DO NOT use test for parameter selection.
    # Cache validation patches once (only ~40 MB at 256x256). Avoid thousands of PNG decodes.
    clean_patches=[ds[i][1] for i in range(len(ds))]
    for sigma in SIGMAS:
        pairs=[(clean.permute(1,2,0).numpy(),
                fixed_noise(clean,sigma,seed,i).permute(1,2,0).numpy())
               for i,clean in enumerate(clean_patches)]
        sums=[]
        for sp,cf in candidates:
            values=[]
            for clean,noisy in pairs:
                filtered=denoise_bilateral(noisy, sigma_color=cf*sigma, sigma_spatial=sp,channel_axis=-1)
                values.append(peak_signal_noise_ratio(clean,filtered,data_range=1.0))
            sums.append((float(np.mean(values)),sp,cf))
        winner=max(sums)
        best[f'sigma_{sigma:.2f}']={'sigma_spatial':winner[1],'sigma_color_multiplier':winner[2]}
        print(f'Bilateral VAL sigma={sigma}: {winner[0]:.3f} dB, spatial={winner[1]}, colorfactor={winner[2]}')
    obj={'protocol':'tuned on DIV2K-valid 0801..0850 ONLY', 'best':best,'seed':seed}
    Path(file_path).write_text(json.dumps(obj,indent=2),encoding='utf8')
    return obj


def load_all_models(paths, device):
    loaded={}
    for label, path in paths.items():
        if path is None or not Path(path).exists():
            print(f'MISSING CHECKPOINT: {label} — skipping; no fabricated scores')
            continue
        checkpoint=safe_torch_load(path)
        if label in ('DnCNN V2','DnCNN V3'):
            model=make_model('v2',checkpoint.get('model_config',{'depth':17,'features':64,'conditioned':True}))
        else:
            model=make_model(checkpoint['architecture'],checkpoint.get('model_config',{}))
        # V3's selected weights may be EMA, while state_dict may be raw.
        state=checkpoint.get('eval_state_dict',checkpoint.get('state_dict'))
        if state is None:raise KeyError(f'{label} checkpoint lacks eval_state_dict / state_dict')
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        loaded[label]=model
        print(f'Loaded {label} ({sum(p.numel() for p in model.parameters()):,} parameters) from {path}')
    return loaded


def lpips_alex(device):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    try:
        return LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval()
    except Exception as exc:
        raise RuntimeError('Cannot initialize AlexNet LPIPS. Enable Kaggle Internet for weights or Add Input with cached weights.') from exc


def measure(pred, clean, perceptual, device):
    pred=pred.float().clamp(0,1).cpu()
    clean=clean.float().cpu()
    a=pred.permute(1,2,0).numpy(); b=clean.permute(1,2,0).numpy()
    scores={'PSNR':float(peak_signal_noise_ratio(b,a,data_range=1.0)),
            'SSIM':float(structural_similarity(b,a,channel_axis=-1,data_range=1.0))}
    if perceptual is None:scores['LPIPS']=None
    else:
        perceptual.reset()
        with torch.inference_mode():
            scores['LPIPS']=float(perceptual(pred[None].to(device),clean[None].to(device)).item())
    return scores


def benchmark(data_root, out_dir, checkpoint_paths, bilateral_file, seed=123, patch_size=256,
              lpips=True, smoke=False, extra_sigma=None, max_images=None):
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=True)
    ds=DIV2KDenoising(data_root,'test',size=patch_size,augment='none')
    if not smoke and len(ds)!=50:
        raise RuntimeError(f'Test must include all 50 unseen images: got {len(ds)}')
    if not smoke and max_images is not None:
        raise ValueError('Limiting test samples is allowed only in smoke mode')
    total=min(len(ds),max_images or len(ds))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained=load_all_models(checkpoint_paths,device)
    perception=lpips_alex(device) if lpips else None
    if bilateral_file is None or not Path(bilateral_file).is_file():
        raise FileNotFoundError('Cannot benchmark without the validation-only bilateral settings')
    tuned=json.loads(Path(bilateral_file).read_text())['best']
    sigmas=list(SIGMAS)
    if extra_sigma is not None:sigmas.append(float(extra_sigma))
    rows, durations = [], defaultdict(lambda:[0.,0])
    panels=[]
    for i in range(total):
        clean=ds[i][1]
        for sigma in sigmas:
            noisy=fixed_noise(clean,sigma,seed,i)
            variants={'Noisy (no processing)':noisy}
            if f'sigma_{sigma:.2f}' in tuned:
                p=tuned[f'sigma_{sigma:.2f}']
                st=time.perf_counter()
                inp=noisy.permute(1,2,0).numpy().astype('float32')
                bilat=denoise_bilateral(inp,sigma_color=p['sigma_color_multiplier']*sigma,
                     sigma_spatial=p['sigma_spatial'],channel_axis=-1)
                variants['Bilateral (skimage)']=torch.from_numpy(np.asarray(bilat,dtype=np.float32).copy()).permute(2,0,1).clamp(0,1)
                durations['Bilateral (skimage)'][0]+=time.perf_counter()-st
                durations['Bilateral (skimage)'][1]+=1
            for label,model in trained.items():
                x=noisy[None].to(device)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.inference_mode():
                    result=model(x,torch.tensor([sigma],device=device)).clamp(0,1)
                if device.type=='cuda':torch.cuda.synchronize()
                durations[label][0]+=time.perf_counter()-start
                durations[label][1]+=1
                variants[label]=result[0].cpu()
            for method,pred in variants.items():
                scores=measure(pred,clean,perception,device)
                rows.append({'image_id':ds.paths[i].stem,'sigma':sigma,'method':method,**scores})
            if i==0 and sigma in SIGMAS:
                panels.append((sigma,clean,{k:v for k,v in variants.items()}))
        if (i+1)%10==0 or (i+1)==total:
            print(f'Benchmark image {i+1}/{total}',flush=True)
    fields=['image_id','sigma','method','PSNR','SSIM','LPIPS']
    with (output/'per_image.csv').open('w',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=fields);dw.writeheader();dw.writerows(rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['sigma'],row['method'])].append(row)
    summary=[]
    for (sigma,method),values in sorted(groups.items()):
        summary.append({'sigma':sigma,'method':method,'images':len(values),
                 **{m:(float(np.mean([v[m] for v in values])) if values[0][m] is not None else None)
                    for m in ('PSNR','SSIM','LPIPS')}})
    with (output/'results.csv').open('w',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=['sigma','method','images','PSNR','SSIM','LPIPS'])
        dw.writeheader();dw.writerows(summary)
    speeds=[]
    for label,(duration,num) in durations.items():
        speeds.append({'method':label,'images':num,'seconds':duration,'images_per_second':num/max(duration,1e-8),
                       'note':'GPU model forward only (single-image)' if label!='Bilateral (skimage)' else 'CPU bilateral filter'})
    with (output/'inference_speed.csv').open('w',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=['method','images','seconds','images_per_second','note'])
        dw.writeheader();dw.writerows(speeds)
    report={'split':'DIV2K_valid_HR 0851..0900 (internal held-out, NOT official DIV2K test)',
            'images':[p.stem for p in ds.paths[:total]],'seed':seed,
            'noise_definition':'V2/V3 fixed_noise(clean,sigma,123,image_index), clipped to [0,1]',
            'patch':patch_size,'lpips':bool(lpips),'methods':list(trained),
            'missing_checkpoints':[name for name,path in checkpoint_paths.items() if path is None or not Path(path).exists()],
            'bilateral_config':str(bilateral_file),'results':summary,'smoke':bool(smoke)}
    (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    md=['# SIGK 2.2 – benchmark porównawczy (DIV2K)',
        '', 'Test wewnętrzny: 0851–0900, 256×256 central crop, identyczny szum dla metod.',
        'Oficjalny baseline: skimage.restoration.denoise_bilateral (parametry dobrane tylko na VAL).',
        'Wyniki **nie** są wynikami oficjalnego testu DIV2K.',
        '','| σ | Metoda | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---|---:|---:|---:|']
    for r in summary:
        md.append(f'| {r["sigma"]:.2f} | {r["method"]} | {r["PSNR"]:.4f} | {r["SSIM"]:.5f} | '+
             (f'{r["LPIPS"]:.5f}' if r['LPIPS'] is not None else 'N/A')+' |')
    md+=['','**Protokół:** modele trenowane wyłącznie na DIV2K train 0001..0800, '
          'dobierane na valid 0801..0850, porównanie na held-out 0851..0900.',
        'Dla pary nowej architektury x1 vs x4 liczba próbek/epok/aktualizacji jest jednakowa.',
        'Budżety starszych V2 i V3 różnią się – tych porównań nie interpretuj jako ablacji augmentacji.',
        'Brakujące modele: '+(', '.join(report['missing_checkpoints']) or 'brak')]
    (output/'report.md').write_text('\n'.join(md),encoding='utf8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if panels:
            sigma,clean,preds=panels[0]
            labels=['Noisy (no processing)','Bilateral (skimage)','DnCNN V2','DnCNN V3',
                    'DRUNet-Lite','ResCrossAttention','DRUNet-Lite ×4','ResCrossAttention ×4']
            picks=[(k,preds[k]) for k in labels if k in preds]
            fig,axes=plt.subplots(1,len(picks)+1,figsize=(3*(len(picks)+1),3.5))
            for ax,(name,image) in zip(axes,[('Clean',clean)]+picks):
                ax.imshow(image.permute(1,2,0).clamp(0,1).numpy())
                ax.set_title(name,fontsize=8);ax.axis('off')
            fig.tight_layout();fig.savefig(output/'example_comparison.png',dpi=120);plt.close(fig)
        if summary:
            for sigma in SIGMAS:
                vals=[r for r in summary if r['sigma']==sigma]
                fig,ax=plt.subplots(figsize=(10,4));ax.bar([r['method'] for r in vals],[r['PSNR'] for r in vals])
                ax.set_title(f'PSNR — sigma {sigma:.2f}');ax.set_ylabel('dB (higher better)')
                ax.tick_params(axis='x',rotation=55);fig.tight_layout()
                fig.savefig(output/f'psnr_{sigma:.2f}.png',dpi=130);plt.close(fig)
    except Exception as exc: print('Visualization optional warning:',repr(exc))
    return summary
