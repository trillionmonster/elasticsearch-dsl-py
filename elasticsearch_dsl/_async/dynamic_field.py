from typing import Any, Collection, Iterable, List, Tuple, Union
from ..utils import AsyncUsingType
from ..dynamic_field_base import SingleFieldBase, SingleFieldValue
from ..async_connections import get_connection
from .mapping import AsyncMapping
from elasticsearch.helpers import async_streaming_bulk
from tqdm.asyncio import tqdm_asyncio
from dataclasses import dataclass

@dataclass
class AsyncSingleField(SingleFieldBase):
    """An async implementation of SingleField for handling single field operations in Elasticsearch.
    
    This class provides asynchronous bulk indexing capabilities for single field values.
    """
    async def put_mapping(self) -> None:
        """
        Initializes the field mapping in Elasticsearch after instance creation.
        Creates or updates the field mapping in the specified index.
        """
        map = AsyncMapping().field(self.field_name, self.field_type)
        await map.save(index=self.index_name)


    async def bulk(
        self,
        values: Iterable[SingleFieldValue],
        using: AsyncUsingType = "default",
        chunk_size: int = 1000,
        total: int = None,
        verbose: bool = False,
        stats_only: bool = False,
        ignore_status: Union[int, Collection[int]] = (),
        *args: Any,
        **kwargs: Any
    ) -> Union[Tuple[int, int], Tuple[int, List]]:
        """Bulk index the given list of values asynchronously.

        Args:
            values: Iterable of values to be indexed
            using: Elasticsearch connection alias
            chunk_size: Size of each bulk indexing batch
            total: Total number of documents (for progress bar)
            verbose: Enable/disable progress bar
            stats_only: If True, return only success/failed counts
            ignore_status: HTTP status codes to ignore
            *args: Additional arguments for streaming_bulk
            **kwargs: Additional keyword arguments for streaming_bulk

        Returns:
            Tuple containing either:
            - (success_count, failure_count) if stats_only=True
            - (success_count, error_list) if stats_only=False

        Raises:
            AssertionError: If values is None or empty
        """
        # Validate input
        assert values, "values cannot be None or empty"

        # Generate bulk actions
        actions_generated = self.generate_dsl_to_bulk(values)
        
        # Get Elasticsearch connection
        es = get_connection(using)

        # Initialize counters and error list
        success, failed = 0, 0
        errors = []

        # Set streaming_bulk parameters
        kwargs.update({
            "yield_ok": True,
            "chunk_size": chunk_size
        })

        # Initialize progress bar if verbose mode is enabled
        pbar = None
        if verbose:
            pbar = tqdm_asyncio(
                desc=f"Bulk Indexing {self.index_name}/{self.field_name}",
                total=total
            )

        try:
            # Process bulk operations
            async for ok, item in async_streaming_bulk(es, actions_generated, ignore_status=ignore_status, *args, **kwargs):
                if not ok:
                    if not stats_only:
                        errors.append(item)
                    failed += 1
                else:
                    success += 1

                # Update progress bar
                if verbose and pbar:
                    pbar.update(1)

        finally:
            # Ensure progress bar is closed
            if verbose and pbar:
                pbar.close()

        # Return results based on stats_only flag
        return (success, failed) if stats_only else (success, errors)