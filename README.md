# Health-Guard

A deep learning based TB (tuberculosis) screening tool that accepts cough 
recordings, chest X-rays, or both, and returns a TB positive/negative result.

## Architecture
- Audio classifier — custom CNN trained on mel spectrograms of cough recordings.
- X-ray classifier — EfficientNet-B0 fine-tuned on chest X-ray images.
- Inference — late fusion via averaged sigmoid outputs when both modalities are present, otherwise single inference.

## Datasets
- Audio: TBScreen Nairobi Dataset.
- X-ray: TBX11K, Montgomery County, Shenzhen, Mendeley Pakistan dataset.

## Usage
```bash
python infer.py --audio patient.wav
python infer.py --xray patient.png
python infer.py --audio patient.wav --xray patient.png
```
## Limitations
- Screening tool only — not a diagnostic system, does not replace clinical judgement.
- Multi-modal fusion is late-stage fusion, with simple averaging, not learned fusion — paired patient data was unavailable for training a proper fusion model.
- Models were rigorously evaluated informally on out-of-distribution data via infer.py, not formal held-out test sets.

## Future Work
- Learned fusion head with paired multi-modal patient data.
- Extension to additional modalities.


The dev branch has implemented this late-stage fusion, ensemble multi-modality.
