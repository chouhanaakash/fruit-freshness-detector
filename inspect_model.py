# inspect_model.py
import h5py
import json
p = "model/fruit_freshness_model.h5"   # adjust if needed

with h5py.File(p, 'r') as f:
    # attributes at root often include 'model_config' or 'keras_version'
    print("Root attrs:", dict(f.attrs))

    # try to read model_config (JSON string)
    if 'model_config' in f.attrs:
        try:
            cfg = json.loads(f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config'])
            print("Found model_config keys:", list(cfg.keys()))
            print("Model class name:", cfg.get('class_name'))
        except Exception as e:
            print("Could not parse model_config:", e)
    # older H5 files may have model_config in group
    if 'keras_version' in f.attrs:
        print("keras_version attribute:", f.attrs['keras_version'])
    # print top-level groups
    print("Top groups:", list(f.keys()))
