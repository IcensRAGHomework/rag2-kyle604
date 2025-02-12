from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)

q1_pdf = "OpenSourceLicenses.pdf"
q2_pdf = "勞動基準法.pdf"


def hw02_1(q1_pdf):
    loader = PyPDFLoader(q1_pdf)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    return chunks[-1]

def hw02_2(q2_pdf):
    loader = PyPDFLoader(q2_pdf)
    docs = loader.load()
    texts = "\n".join(doc.page_content for doc in docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=5, separators=["\n *第 "], is_separator_regex=True)
    chunks = text_splitter.split_text(texts)
    return len(chunks)

if __name__ == "__main__":
    #print(hw02_1(q1_pdf))
    print(hw02_2(q2_pdf))
