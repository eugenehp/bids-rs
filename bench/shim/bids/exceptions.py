"""Exception hierarchy mirroring ``bids.exceptions``."""


class PyBIDSError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(ValueError, PyBIDSError):
    """Raised on an invalid configuration."""


class NoMatchError(ValueError, PyBIDSError):
    """Raised when a query matches nothing and a match was required."""


class BIDSEntityError(AttributeError, PyBIDSError):
    """Raised on an unknown entity."""


class TargetError(ValueError, PyBIDSError):
    """Raised when ``return_type`` needs a ``target`` and none was given."""


class BIDSValidationError(ValueError, PyBIDSError):
    """Raised when a dataset fails validation."""


class BIDSDerivativesValidationError(BIDSValidationError):
    """Raised when a derivatives dataset fails validation."""


class BIDSConflictingValuesError(BIDSValidationError):
    """Raised when a file carries conflicting entity values."""


class BIDSChildDatasetError(ValueError, PyBIDSError):
    """Raised when a child dataset is used where a root was expected."""
