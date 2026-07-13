from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def code_chunker(max_chunk_size=2000):
    loader = DirectoryLoader(
        path="data/raw/",
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
    print(f"Total Python code chunks created: {len(code_chunks)}")
    print(code_chunks[0])
