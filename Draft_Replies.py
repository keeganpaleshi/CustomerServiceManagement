import warnings

warnings.warn(
    "Draft_Replies has been retired because Gmail draft workflows are no longer supported. "
    "Use utils.generate_ai_reply for non-Gmail reply generation.",
    DeprecationWarning,
    stacklevel=2,
)
