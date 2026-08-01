from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..data import MinimalSource


def text_chunker(
    repo_path: str = "data/raw/",
    max_chunk_size: int = 2000,
) -> list[MinimalSource]:
    """Chunk every markdown file under repo_path into MinimalSource spans.

    Args:
        repo_path: Directory to recursively search for .md files.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        One MinimalSource per chunk, with character offsets into the
        original file content.
    """
    loader = DirectoryLoader(
        path=repo_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        recursive=True,
        silent_errors=True,
    )
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )

    final_chunks = text_splitter.split_documents(raw_docs)

    return [
        MinimalSource(
            chunk_content=chunk.page_content,
            file_path=chunk.metadata["source"],
            first_character_index=chunk.metadata["start_index"],
            last_character_index=(
                chunk.metadata["start_index"] + len(chunk.page_content)
            ),
        )
        for chunk in final_chunks
    ]
