try:
    import streamlit 
    from .app import run_ui
except ImportError as e:
    raise ImportError(
        "The RapidSegment UI requires the 'ui' extra.\n"
        "Install it with:  pip install 'rapidsegment[ui]'"
    ) from e

__all__ = ["run_ui"]