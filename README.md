<div dir="rtl">

# بازتولید Data Audit و Region-Query Pilot روی MosMedData+ و QaTa-COV19

هدف این repository این است که هر کسی فقط با تنظیم مسیر دیتاست‌ها بتواند **Data Audit** و **Pilot** را دوباره اجرا کند.
گزارش اجرای نویسنده، همراه با نتایج و visualizationها برای بخش‌های **RQ** و **text association**، جداگانه در اختیار شما قرار می‌گیرد.

---

## 1. ساختار پروژه

```
project/
├── README.md
├── requirements.txt
├── configs/
│   ├── mosmed_rq.yaml          # config آزمایش MosMed_RQ.py
│   ├── mosmed_rq_ta.yaml      # config آزمایش MosMed_RQ_TextAssocciation.py
│   └── qata_rq.yaml            # config آزمایش QaTa_RQ.py
├── splits/                     # split manifestهای نهایی
│   ├── mosmed/                 # Train_resplit_by_group.csv, Val_resplit_by_group.csv, ...
│   └── qata/                   # Train_ID(sheet1).csv
├── scripts/
│   ├── MosMedData_Resplit.py
│   ├── MosMed_threshold.py                       # MosMed Data Audit + انتخاب threshold (224x224)
│   ├── MosMed_K_Audit.py
│   ├── QaTa.py                         # QaTa Data Audit + انتخاب threshold
│   ├── MosMed_RQ.py
│   ├── MosMed_RQ_TextAssocciatin.py   # Pilot + text association / perturbation
│   └── QaTa_RQ.py
├── data/                       # دیتاست‌ها (در repository نیست؛ بخش 3)
└── outputs/                    # خروجی‌ها (هنگام اجرا ساخته می‌شود)
```

همهٔ دستورها را از **ریشهٔ پروژه** اجرا کنید (پوشه‌ای که `README.md` در آن است).

---

## 2. نصب

```bash
pip install -r requirements.txt
```

- نسخه‌ها مطابق محیط اجرای اصلی (Kaggle، Python 3.13) پین شده‌اند.
- `torch` با CUDA خاصی ساخته شده است. اگر CUDA شما فرق دارد، ابتدا نسخهٔ مناسب `torch` را از سایت pytorch.org نصب کنید و بعد دستور بالا را اجرا کنید.
- برای Pilotها GPU توصیه می‌شود.
- مدل‌های pretrained به‌صورت پیش‌فرض از مسیر `./lib/BiomedVLP-CXR-BERT-specialized` و `./lib/convnext-tiny-224` خوانده می‌شوند. اگر این پوشه‌ها وجود نداشته باشند، کد به‌طور خودکار `microsoft/BiomedVLP-CXR-BERT-specialized` و `facebook/convnext-tiny-224` را از Hugging Face دانلود می‌کند (به اینترنت نیاز دارد).

---

## 3. دیتاست‌ها

| دیتاست | محتوا | منبع اصلی (Kaggle) |
|---|---|---|
| MosMedData+ | frames و masks | `maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset` |
| MosMedData+ text | فایل‌های متن train/val/test اصلی | `mohaddesegh/mosmeddata` |
| QaTa-COV19-v2 | تصاویر و Ground-truthها | `aysendegerli/qatacov19-dataset` |
| QaTa text | `Train_text_for_Covid19(Sheet1).csv` | `mohaddesegh/qatadataset` |
| QaTa split | `Train_ID(sheet1).csv` | `mohaddesegh/qatadataset-split` |

ساختار پیش‌فرض پوشه‌ها (اگر دیتاها را همین‌جا بگذارید، به هیچ آرگومانی نیاز نیست):

```
data/
├── MosMedData/
│   ├── frames/
│   ├── masks/
│   └── text/
│       ├── Train_text_MosMedData 1(in).csv
│       ├── Val_text_MosMedData 1(in).csv
│       └── Test_text_MosMedData(in).csv
└── QaTa-COV19/
    ├── QaTa-COV19-v2/
    │   └── Train Set/
    │       ├── Images/
    │       └── Ground-truths/
    └── text/
        └── Train_text_for_Covid19(Sheet1).csv
```

split manifestهای نهایی (CSVهای مرحلهٔ 1 و `Train_ID(sheet1).csv`) داخل `splits/` قرار دارند.

---

## 4. تنظیم مسیرها (قبل از اجرا)

**فقط مسیر دیتاست‌ها را عوض کنید؛ به خود کدها دست نزنید.**

**Pilotها** (`MosMed_RQ_clean.py`، `MosMed_RQ_v11_reproducible_clean.py`، `QaTa_RQ_clean.py`):
مسیرها در فایل‌های `configs/*.yaml` هستند (بخش‌های `DATA` و `PILOT.output_dir`). همان فایل را ویرایش کنید، یا مسیر را هنگام اجرا بدهید:

```bash
python scripts/MosMed_RQ_TextAssocciation.py \
    --config configs/mosmed_rq_ta.yaml \
    --frames_dir /my/data/frames --masks_dir /my/data/masks
```

آرگومان‌های قابل override در Pilotها: `--frames_dir`، `--masks_dir`، `--train_csv`، `--valid_csv`، `--output_dir`، `--bert_type`، `--vision_type` (برای QaTa: `--dataset_root` و `--train_id_csv` هم).

**Resplit و Auditها** (`MosMedData_Resplit.py`، `MosMed_threshold.py`، `MosMed_K_Audit.py`، `QaTa.py`):
مسیرها آرگومان خط فرمان هستند و پیش‌فرضشان همان ساختار بخش 3 است. برای دیدن همهٔ آرگومان‌ها:

```bash
python scripts/MosMed_threshold.py --help
```

---

## 5. ترتیب اجرا

### 1) Resplit (MosMedData+)

```bash
python scripts/MosMedData_Resplit.py
```

re-split مبتنی بر group و source-stratified را اجرا می‌کند.
خروجی‌ها در `outputs/MosMedData_Resplit_SourceStratified/`: `Train_resplit_by_group.csv`، `Val_resplit_by_group.csv`، `Test_resplit_by_group.csv`، `split_manifest.json`، `resplit_verification_report.txt`.
اگر می‌خواهید با split ثبت‌شدهٔ نویسنده کار کنید، همان فایل‌های داخل `splits/mosmed/` را استفاده کنید. در غیر این صورت CSVهای تولیدشده را در `splits/mosmed/` کپی کنید.

### 2) MosMed Data Audit و انتخاب threshold (224x224)

```bash
python scripts/MosMed_threshold.py
```

ماسک‌ها را به 224x224 تبدیل می‌کند، connected componentها را استخراج می‌کند و threshold را فقط از روی train (بدون استفاده از متن) انتخاب می‌کند.
خروجی مهم: `threshold_selection_metadata.json`، `frozen_threshold_train_validation_summary.csv`، `04_Data_Audit_Raw_Results.xlsx`، `AUDIT_METHODS.txt`.

### 3) MosMed K Audit

```bash
python scripts/MosMed_K_Audit.py
```

خروجی: `MosMed_K_Audit_Summary.csv`، `MosMed_K_Audit_Metadata.json`.

### 4) QaTa Data Audit

```bash
python scripts/QaTa.py
```

خروجی مهم: `threshold_selection_metadata.json`، `frozen_threshold_train_validation_summary.csv`، `AUDIT_METHODS.txt`.

### 5) MosMed Region-Query Pilot

```bash
python scripts/MosMed_RQ.py --config configs/mosmed_rq.yaml
```

خروجی در `outputs/MosMed_RQ_seed27/`: `pilot_config.json`، `pilot_history.csv`، `pilot_final_summary.json`، `pilot_per_image_diagnostics.csv` و checkpointها.

### 6) MosMed Region-Query Pilot و آزمون text association (v11)

```bash
python scripts/MosMed_RQ_TextAssocciation.py --config configs/mosmed_rq_ta.yaml
```

علاوه بر آموزش Pilot، آزمون text perturbation / association را اجرا می‌کند و `text_perturbation_*.csv`، `text_association_weights.csv`، `orientation_audit.json` و `raw_spatial_attention_audit.csv` را تولید می‌کند.
اگر پوشهٔ خروجی از قبل وجود داشته باشد، اجرای جدید در پوشه‌ای با پسوند `_run2`، `_run3`، ... ذخیره می‌شود و نتیجهٔ قبلی بازنویسی نمی‌شود.

### 7) QaTa Region-Query Pilot

```bash
python scripts/QaTa_RQ.py --config configs/qata_rq.yaml
```

---

## 6. بازتولیدپذیری

- seed همهٔ آزمایش‌ها در config برابر `27` است (`seed` و `data_seed`).
- نسخهٔ v11 الگوریتم‌های deterministic را فعال می‌کند. با این حال روی GPU یا نسخهٔ CUDA متفاوت ممکن است اختلاف‌های عددی کوچک دیده شود.
- نام پوشهٔ خروجی Pilotها از `output_dir` به‌اضافهٔ `_seed27` ساخته می‌شود.
- پیام‌های خطای مسیر به آرگومان یا فایل config اشاره می‌کنند.

---

## 7. نتایج مرجع

گزارش اجرای نویسنده، به‌همراه نتایج و visualizationها برای **RQ** و **text association**، جداگانه ارائه می‌شود تا نتایج اجرای شما با آن مقایسه شود.

</div>
