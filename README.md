## HAComb

Code for our submission ICME-2025 #477 paper

### Data Source

MDD-12: MTDiag: an effective multi-task framework for automatic diagnosis (AAAI 2023)

MZ-10: DxFormer: a decoupled automatic diagnostic system based on decoder--encoder transformer with dense symptom representations (Bioinformatics 2022)

Dxy-5: End-to-end knowledge-routed relational dialogue system for automatic diagnosis (AAAI 2019)

MZ-4: Task-oriented dialogue system for automatic diagnosis (ACL 2018)

### Environment Setup:
```yaml
conda env create --name HAComb --file environment.yml
```

### Train Ours (HAComb)

```yaml
python train_hac.py \
-d MDD12 
```


