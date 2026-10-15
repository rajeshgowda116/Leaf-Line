import os
import threading

_CACHED_MODEL = None
_LOCK = threading.Lock()

def get_cached_keras_model(model_path: str):
    """Safely returns cached Keras model instance."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    with _LOCK:
        if _CACHED_MODEL is not None:
            return _CACHED_MODEL

        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras

        # Disable torch dynamo compiler if running with torch backend (prevents missing MSVC compiler error on Windows)
        try:
            import torch
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.disable = True
        except Exception:
            pass

        model = keras.models.load_model(model_path)
        try:
            model.compile(jit_compile=False)
        except Exception:
            pass

        _CACHED_MODEL = model
        return _CACHED_MODEL
