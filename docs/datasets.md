# Dataset access and licence status

Checked 24 August 2026. **Licence status is as important as availability here:**
a dataset that cannot be used commercially can validate the pipeline but cannot
train the shipping model.

Rows marked *(verified)* were read directly from the source page or licence
file. Rows marked *(reported)* come from search summaries and should be
confirmed against the source before anyone relies on them.

| Dataset | Access | Commercial use | Notes |
|---|---|---|---|
| **MCD-rPPG** | direct download, no form *(verified)* | **Yes -- CC-BY-4.0** *(verified)* | 600 real subjects, rest + post-exercise, PPG + ECG, consumer cameras |
| **SCAMPS** | direct download, no form *(verified)* | **No** *(verified)* | Synthetic avatars. R-UDA research-only |
| **UBFC-rPPG** | email request *(reported)* | request-dependent | Uncompressed 8-bit RGB, 30 fps |
| **UBFC-Phys** | IEEE DataPort, open access *(reported)* | request-dependent | Video + BVP + EDA + task labels |
| **PURE** | email request *(reported)* | request-dependent | The motion-robustness benchmark |
| **iBVP** | signed EULA *(verified)* | **No**, academic only *(verified)* | ~400 GB. RGB + thermal |
| **MMPD** | request *(reported)* | **No**, commercial banned *(reported)* | Mobile phone video |
| **LGI-PPGI** | **effectively unavailable** *(reported)* | -- | Hosting withdrawn, no downloads |
| **VitalVideos** | request *(reported)* | **Yes, for a fee** *(reported)* | The only commercial path found |

Contacts, where a request is needed:

- UBFC-rPPG -- yannick.benezeth@u-bourgogne.fr
  (https://sites.google.com/view/ybenezeth/ubfcrppg)
- PURE -- nikr-datasets-request@tu-ilmenau.de
- UBFC-Phys -- https://ieee-dataport.org/open-access/ubfc-phys-2

## Correction to an earlier conclusion in this file

An earlier version of this document stated flatly that **no public rPPG dataset
permits commercial use for free**, and drew the strategic conclusion that public
data could only ever be used for validation. That was wrong, and it was wrong in
the expensive direction: it would have justified skipping a search that turned
out to succeed.

**MCD-rPPG is CC-BY-4.0** -- 600 real subjects, contact PPG and ECG, three
consumer cameras, rest and post-exercise, on Hugging Face with no form, no EULA
and no account. Attribution is the only obligation.

The corrected position:

- **A commercially usable public dataset exists**, and it is a better fit for
  this product than the research-only ones: consumer cameras, compressed video,
  and a working population rather than a lab cohort.
- **It still does not replace the project's own data.** Design doc section 6 is
  right that the moat is consented, domain-specific data from pilots. MCD-rPPG
  is a starting point for the sensor layer, not a substitute for data about the
  customers' actual sessions and tasks.
- **Research-only datasets remain validation-only.** SCAMPS, iBVP and MMPD can
  check that algorithms behave; they cannot train anything that ships.

### On third-party mirrors

Hugging Face also hosts re-uploads of UBFC-rPPG (`WeiQian98/UBFC-rPPG`,
`Horusprg/UBFC-rPPG-Faces`). **These are not a way around the access process.**
Re-uploading a dataset does not grant rights the uploader never had, so the
original terms still apply and the mirror carries no licence of its own. They
are not used here.

## SCAMPS in particular

Downloaded: `scamps_videos_example.tar.gz`, 1.14 GiB, 10 clips, from
`facesyntheticspubwedata.z6.web.core.windows.net` (the URL published in the
official repository, https://github.com/danmcduff/scampsdataset).

The licence restriction is broad -- it covers using the data or any results
"to improve any product or service" -- so the safe reading is: use it to
verify that published algorithms (POS, CHROM) behave correctly on realistic
faces, and do not train any proprietary model on it. Where the boundary sits
for engineering validation of a commercial product is a question for counsel,
not for this file.

### What SCAMPS can answer that the built-in synthetic generator cannot

The generator in `training/datasets/synthetic.py` paints a pulse into a flat
ellipse. It validates the harness arithmetic and nothing about faces. SCAMPS
renders photorealistic avatars, so it is the first data in this project that
exercises:

- face detection and landmark-free ROI selection on real facial geometry
- skin rendering with subsurface scattering, rather than a flat colour patch
- genuine head pose variation (`d_pitch`, `d_yaw`, `d_rol` are provided)
- facial action units, i.e. expression-driven non-rigid deformation

### What it still cannot answer

It is rendered, not recorded. It does not settle real skin tone response, real
camera pipelines, real ambient lighting, or real motion artefacts. The HR MAE
go/no-go bar in the design doc remains a question for UBFC-rPPG and PURE.
