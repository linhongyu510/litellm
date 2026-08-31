import pytest

from litellm.rag.text_splitters.recursive_character_text_splitter import (
    RecursiveCharacterTextSplitter,
)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "error"),
    [
        (0, 0, "chunk_size must be greater than 0"),
        (-1, 0, "chunk_size must be greater than 0"),
        (100, -1, "chunk_overlap must be non-negative"),
        (100, 100, "chunk_overlap must be smaller than chunk_size"),
        (100, 101, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_rejects_invalid_chunking_configuration(
    chunk_size: int,
    chunk_overlap: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
