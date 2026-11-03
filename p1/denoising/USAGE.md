# Denoising 2.2

Uruchomienie z katalogu głównego repozytorium:

```bash
python -m pip install -r p1/denoising/requirements.txt
bash p1/scripts/download_data.sh
python -m p1.denoising.cli benchmark
```

Trening:

```bash
python -m p1.denoising.cli train-v2
python -m p1.denoising.cli train-v3
python -m p1.denoising.cli train-v4 --variant all --multi-gpu
python -m p1.denoising.cli train-v4 --variant rescross_x4 --batch-size 32
```

Baseline:

```bash
python -m p1.denoising.cli tune-baseline
python -m p1.denoising.cli benchmark --retune
```

Dane: `p1/data/raw/DIV2K/train`, `p1/data/raw/DIV2K/valid`.
Walidacja: 0801–0850. Test wewnętrzny: 0851–0900.
Wagi: `p1/checkpoints/denoising/`. Wyniki uruchomień: `p1/output/denoising/`.
Modele testowane przy σ = 0.01 i σ = 0.03.

Aby wznowić V4, uruchom ten sam wariant i parametry w tym samym katalogu wyników.
Do ewaluacji bez treningu używane są dołączone pliki `best.pt`.
