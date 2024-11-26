from typing import Any, Collection, Iterable, Union, Tuple, List
from ..dynamic_field_base import SingleFieldBase, SingleFieldValue
from ..utils import UsingType
from ..connections import get_connection
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm
from dataclasses import dataclass
from .mapping import Mapping

@dataclass
class SingleField(SingleFieldBase):
    """A class representing a single field in Elasticsearch with bulk indexing capabilities.
    
    Inherits from SingleFieldBase to handle single field operations in Elasticsearch.
    """
    def __post_init__(self) -> None:
        """
        Initializes the field mapping in Elasticsearch after instance creation.
        Creates or updates the field mapping in the specified index.
        """
        Mapping().field(self.field_name, self.field_type).save(index=self.index_name)

    def bulk(
        self,
        values: Iterable[SingleFieldValue],
        using: UsingType = "default",
        chunk_size: int = 1000,
        total: int = None,
        verbose: bool = False,
        stats_only: bool = False,
        ignore_status: Union[int, Collection[int]] = (),
        *args: Any,
        **kwargs: Any
    ) -> Union[Tuple[int, int], Tuple[int, List]]:
        """Bulk index the given list of values into Elasticsearch.

        This method efficiently processes and indexes multiple field values using Elasticsearch's bulk API.

        Args:
            values: An iterable of values to be bulk indexed
            using: Elasticsearch connection alias to use
            chunk_size: Number of documents to send in each bulk request
            total: Total number of documents (for progress tracking)
            verbose: Whether to display a progress bar
            stats_only: If True, return only success/failure counts
            ignore_status: HTTP status codes to ignore during indexing
            *args: Additional arguments for streaming_bulk
            **kwargs: Additional keyword arguments for streaming_bulk

        Returns:
            If stats_only is True:
                A tuple of (successful_operations, failed_operations)
            If stats_only is False:
                A tuple of (successful_operations, error_list)

        Raises:
            AssertionError: If values is None or empty
        """
        # Validate input
        assert values, "value_list cannot be None or empty"
        
        # Generate bulk actions
        actions_generated = self.generate_dsl_to_bulk(values)
        
        # Get Elasticsearch connection
        es = get_connection(using)

        success, failed = 0, 0
        errors = [] if not stats_only else None

        # Initialize progress bar if verbose
        if verbose:
            pbar = tqdm(
                desc=f"Bulk Indexing {self.index_name}/{self.field_name}",
                total=total,
                unit='docs'
            )

        try:
            # Process bulk operations
            for ok, item in streaming_bulk(
                    es,
                    actions_generated,
                    ignore_status=ignore_status,
                    span_name="helpers.bulk",
                    chunk_size=chunk_size,
                    yield_ok=True,
                    *args,
                    **kwargs
            ):
                if not ok:
                    if not stats_only:
                        errors.append(item)
                    failed += 1
                else:
                    success += 1

                if verbose:
                    pbar.update(1)

        finally:
            # Ensure progress bar is closed
            if verbose:
                pbar.close()

        return (success, failed) if stats_only else (success, errors)
