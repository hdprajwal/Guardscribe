import kagglehub

kagglehub.login()

# Upload a model to Kaggle Hub.
LOCAL_MODEL_DIR = [
    './toxic_content_detector_model',
    './toxic_span_distilbert'
]

MODEL_SLUG = [
    'toxic-content-detector',
    'toxic-span-distilbert'
]

VARIATION_SLUG = 'default'

for model_path, model_slug in zip(LOCAL_MODEL_DIR, MODEL_SLUG):
    kagglehub.model_upload(
        handle=f"hdprajwal/{model_slug}/transformers/{VARIATION_SLUG}",
        local_model_dir=model_path,
        version_notes='Update 2025-04-30')
