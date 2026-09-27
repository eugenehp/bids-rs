"""Indexer options mirroring ``bids.layout.index``."""

__all__ = ['BIDSLayoutIndexer']


class BIDSLayoutIndexer:
    """Indexing options for a :class:`~bids.layout.BIDSLayout`.

    Holds the same options PyBIDS' indexer does; the layout reads them when it
    builds its index.
    """

    def __init__(
        self,
        validate=True,
        ignore=None,
        force_index=None,
        index_metadata=True,
        config_filename='layout_config.json',
        **filters,
    ):
        self.validate = validate
        self.ignore = ignore
        self.force_index = force_index
        self.index_metadata = index_metadata
        self.config_filename = config_filename
        self.filters = filters

    def __repr__(self):
        return (
            f'BIDSLayoutIndexer(validate={self.validate}, '
            f'index_metadata={self.index_metadata})'
        )
