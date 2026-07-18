from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ..data import MinimalSource


def code_chunker(
    repo_path: str = "data/raw/",
    max_chunk_size: int = 2000,
) -> list[tuple[str, MinimalSource]]:
    loader = DirectoryLoader(
        path=repo_path,
        glob="**/*.py",
        loader_cls=TextLoader,
        recursive=True
    )
    raw_docs = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )

    code_chunks = python_splitter.split_documents(raw_docs)

    return [
        (
            chunk.page_content,
            MinimalSource(
                file_path=chunk.metadata["source"],
                first_character_index=chunk.metadata["start_index"],
                last_character_index=(
                    chunk.metadata["start_index"] + len(chunk.page_content)
                ),
            ),
        )
        for chunk in code_chunks
    ]
