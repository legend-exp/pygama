from __future__ import annotations


class DSPError(Exception):
    """Base class for signal processors."""

    pass


class DSPFatal(DSPError):
    """Fatal error thrown by DSP processors that halts production.

    Attributes
    ----------
    wf_range: range
        range of wf indices. This will be set after the exception is caught,
        and appended to the error message
    processor: str
        string of processor and arguments. This will be set after the exception
        is caught, and appended to the error message
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.wf_range = None
        self.processor = None

    def __str__(self) -> str:
        suffix = ""
        if self.wf_range:
            suffix += "\nThrown while processing entries " + str(self.wf_range)
        if self.processor:
            suffix += "\nThrown by " + self.processor
        return super().__str__() + suffix


class ProcessingChainError(DSPError):
    """Error thrown when there is a problem setting up a processing chain."""

    pass
