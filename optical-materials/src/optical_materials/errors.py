class MaterialError(ValueError):
    """A material definition or response is invalid."""
class RangeError(MaterialError):
    """A requested frequency lies outside the material's valid range."""
class ResponseUnavailableError(MaterialError):
    """The requested response needs additional physical information."""
class MaterialWarning(UserWarning):
    """Input was repaired in an explicitly reported way."""