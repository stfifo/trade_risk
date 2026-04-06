import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

print("=== reuter news Vector DB(ChromaDB)===")

news_df = pd.read_csv('../data/clean_reuters.csv', encoding='utf-8-sig')
news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
news_df = news_df.dropna(subset=['date', 'main_text']).copy()

# text chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

documents = []
for _, row in news_df.iterrows():
    chunks = text_splitter.split_text(str(row['main_text']))
    
    # datatime -> int(YYYYMM)
    date_int = int(row['date'].strftime('%Y%m'))
    
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "date_int": date_int, # 나중에 $lte(이하) 필터링에 사용
                "date_str": row['date'].strftime('%Y-%m-%d'),
                "title": str(row['title'])
            }
        )
        documents.append(doc)

# print(f"total {len(documents)} doc chunk. start enbedding and store vectorDB...")

# embedding & vectorDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="../data/chroma_db"
)

# print("ChromaDB complete")