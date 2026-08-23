try:
    import streamlit  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The RapidSegment UI requires the 'ui' extra.\n"
        "Install it with:  pip install 'rapidsegment[ui]'"
    ) from e


def __getattr__(name):
    """Lazily expose ``run_ui`` to avoid a circular import with ``app.py``.

    ``app.py`` (the Streamlit entry point) imports sibling modules through the
    ``rapidsegment.ui`` package. Eagerly importing ``.app`` here would re-enter
    ``app.py`` while it is still initialising (it imports ``_exit``/``_theme``,
    both of which trigger this package's import) and deadlock under Streamlit's
    multipage loader. Lazily resolving ``run_ui`` keeps
    ``from rapidsegment.ui import run_ui`` working for the CLI while breaking
    the cycle.
    """
    if name == "run_ui":
        from .app import run_ui
        return run_ui
    raise AttributeError(f"module 'rapidsegment.ui' has no attribute {name!r}")


__all__ = ["run_ui"]