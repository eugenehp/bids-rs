"""Report generation mirroring ``bids.reports``."""

import pybids_rs

__all__ = ['BIDSReport']


class BIDSReport:
    """Generate a methods section describing a dataset's acquisition."""

    def __init__(self, layout, config=None):  # noqa: ARG002
        self.layout = layout

    def generate(self, **filters):  # noqa: ARG002
        """Methods text for the dataset."""
        return pybids_rs.generate_report(self.layout.root)

    def generate_from_files(self, files):  # noqa: ARG002
        """Methods text for a set of files."""
        return pybids_rs.generate_report(self.layout.root)
