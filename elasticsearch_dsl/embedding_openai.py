from typing import Iterable, List
import openai
import numpy as np
from tqdm import tqdm


def l2_normalize(vectors) -> np.ndarray:
    """Normalize vectors using L2 normalization.
    
    Args:
        vectors: Input vectors to normalize
        
    Returns:
        np.ndarray: L2 normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, np.finfo(float).eps)
    return vectors / norms


class Embedding:
    """A class to generate embeddings using OpenAI's API.
    
    This class provides methods to generate embeddings for text using OpenAI's
    embedding models, with support for batch processing and streaming.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_token: str = None,
        select_dims: int = 0
    ) -> None:
        """Initialize the Embedding class.
        
        Args:
            base_url: The base URL for the OpenAI API
            model_name: Name of the embedding model to use
            api_token: OpenAI API token (optional)
            select_dims: Number of dimensions to select from the embedding (0 means all)
        """
        self.model_name = model_name
        self.select_dims = select_dims
        self.client = openai.Client(
            api_key=api_token or "blank", 
            base_url=base_url
        )

    def embedding(
            self,
            texts: List[str],
            batch_size: int = 32,
            verbose: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            verbose: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        n = len(texts)
        embeddings = []
        
        if verbose:
            pbar = tqdm(desc="Embedding", total=len(texts))

        # Process texts in batches
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            batch_result = self.client.embeddings.create(
                model=self.model_name, 
                input=batch
            ).data
            embeddings.append(
                np.array([item.embedding for item in batch_result])
            )
            if verbose:
                pbar.update(len(batch))

        # Concatenate and process embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        if self.select_dims > 0:
            embeddings = embeddings[:, :self.select_dims]
        embeddings = l2_normalize(embeddings)

        return embeddings.tolist()
    
    def iter_embedding(
        self,
        texts_generator: Iterable[str],
        batch_size: int = 32,
        verbose: bool = False,
        total: int = None
    ) -> Iterable[List[float]]:
        """Generate embeddings for texts in a streaming fashion.
        
        Args:
            texts_generator: Iterator yielding texts
            batch_size: Number of texts to process in each batch
            verbose: Whether to show progress bar
            total: Total number of texts (for progress bar)
            
        Yields:
            Embedding vectors one at a time
        """
        batch = []
        if verbose:
            pbar = tqdm(desc="Embedding", total=total)
            
        # Process texts in batches
        for text in texts_generator:
            batch.append(text)
            if len(batch) >= batch_size:
                embeddings = self.embedding(batch, batch_size=batch_size)
                if verbose:
                    pbar.update(len(batch))
                yield from embeddings
                batch = []
                
        # Process remaining texts
        if batch:
            embeddings = self.embedding(batch, batch_size=batch_size)
            if verbose:
                pbar.update(len(batch))
            yield from embeddings
