from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_chunker(max_chunk_size=2000):
    loader = DirectoryLoader(
        path="data/raw/",
        glob="**/*.md",
        loader_cls=TextLoader,
        recursive=True
    )
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )

    final_chunks = text_splitter.split_documents(raw_docs)

    print(f"Total chunks created: {len(final_chunks)}")
    print(final_chunks[0])
