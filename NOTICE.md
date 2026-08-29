# Third-party code, models and data

## Vendored source under the Responsible AI Source Code License (RAIL)

`neuroproxy/rppg/neural/efficientphys.py` is adapted from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
(`neural_methods/model/EfficientPhys.py`), which implements

> Liu et al., *EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based
> Cardiac Measurement*, WACV 2023.

That code is licensed under the **Responsible AI Source Code License v1.1**.
Commercial use is permitted. Behavioural restrictions are not, and **section
3.2 of that licence requires them to be passed through contractually to every
downstream user**.

The restricted uses include, in the licence's own terms:

- surveillance, and inferring identity attributes including health and medical
  conditions;
- determining insurance premiums, or denying insurance applications or claims;
- diagnosing a medical condition without human oversight;
- predicting criminal behaviour from personal characteristics, explicitly
  including **heart rate, perspiration and breathing**.

**Anyone shipping a product derived from this repository must carry these
restrictions in their own terms of service.** They overlap almost exactly with
the exclusions this project sets for itself; the obligation that is easy to
miss is the pass-through, not the restrictions themselves.

## Pretrained weights

Not redistributed here. `scripts/fetch_models.sh` downloads them from the
rPPG-Toolbox release directory, under the same licence.

## Datasets

None are redistributed. Access routes and licence status for each are in
[docs/datasets.md](docs/datasets.md). In summary:

| dataset | access | commercial use |
|---|---|---|
| MCD-rPPG | direct download, no form | **yes, CC-BY-4.0** |
| SCAMPS | direct download, no form | no, research only (R-UDA) |
| UBFC-rPPG, UBFC-Phys, PURE | request required | request-dependent |

Benchmark numbers produced from research-licensed data are validation results,
not product claims, and are labelled as such throughout `docs/limitations.md`.
