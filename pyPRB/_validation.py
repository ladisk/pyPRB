"""Private input checks shared by the modules of pyPRB."""


def _positive(**values) -> None:
    """Raise ValueError unless every keyword value is positive."""
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")
