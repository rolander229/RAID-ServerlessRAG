import asyncio
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import asyncpg
from pgvector.asyncpg import register_vector
import hashlib
from vllm import LLM, SamplingParams
import argparse
import os

class MonolithicRAG:
    def __init__(self, embed_model_name, rerank_model_name, llm_model_name, db_config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        self.embed_model.half()
        
        print(f"Loading reranker model: {rerank_model_name}")
        self.rerank_model = CrossEncoder(rerank_model_name, device=self.device)
        self.rerank_model.model.half()
        
        print(f"Loading LLM model: {llm_model_name}")
        # Note: vLLM offline inference
        self.llm = LLM(
            model=llm_model_name,
            trust_remote_code=False,
            dtype="float16",
            gpu_memory_utilization=0.8,
            enforce_eager=True
        )
        
        self.db_config = db_config
        self.pool = None
        self.vector_dim = 384 # Hardcoded based on bge-small-en-v1.5
        
    async def init_db(self):
        self.pool = await asyncpg.create_pool(**self.db_config)
        async with self.pool.acquire() as conn:
            await register_vector(conn)

    def _get_table_name(self, query_id: str) -> str:
        hash_obj = hashlib.md5(query_id.encode())
        return f"monolith_vectors_{hash_obj.hexdigest()}"

    async def ensure_table(self, table_name: str):
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({self.vector_dim}),
                    text TEXT
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                ON {table_name} USING hnsw (embedding vector_cosine_ops)
            """)

    async def ingest(self, table_name, passages):
        print(f"Ingesting {len(passages)} passages...")
        embeddings = self.embed_model.encode(passages, convert_to_numpy=True)
        
        async with self.pool.acquire() as conn:
            data = [(emb, txt) for emb, txt in zip(embeddings, passages)]
            await conn.copy_records_to_table(
                table_name, records=data, columns=["embedding", "text"]
            )
        return True

    async def search(self, table_name, query_text, top_k=20):
        query_embedding = self.embed_model.encode([query_text], convert_to_numpy=True)[0]
        
        async with self.pool.acquire() as conn:
            # Cosine similarity search using pgvector
            results = await conn.fetch(
                f"""
                SELECT text FROM {table_name}
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                top_k
            )
            return [r["text"] for r in results]

    def rerank(self, query, passages, top_k=2):
        print(f"Reranking {len(passages)} candidates...")
        pairs = [[query, p] for p in passages]
        scores = self.rerank_model.predict(pairs)
        
        # Sort by score descending
        scored_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return [p for p, s in scored_passages[:top_k]]

    def generate(self, query, context):
        prompt_template = "Question: {question}\nContext: {context}\nAnswer:"
        prompt = prompt_template.format(question=query, context="\n".join(context))
        
        sampling_params = SamplingParams(max_tokens=100, temperature=0)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    async def run_pipeline(self, passages, queries):
        query_id = "single_session"
        table_name = self._get_table_name(query_id)
        
        await self.ensure_table(table_name)
        await self.ingest(table_name, passages)
        
        results = []
        for q in queries:
            print(f"\nProcessing query: {q}")
            search_results = await self.search(table_name, q, top_k=20)
            reranked_results = self.rerank(q, search_results, top_k=2)
            answer = self.generate(q, reranked_results)
            results.append({
                "query": q,
                "answer": answer
            })
        return results

async def main():
    db_config = {
        "host": "localhost",
        # "host": "pg-vector-service",
        "port": 5432,
        "user": "asplos25",
        "password": "123456",
        "database": "database_asplos",
    }
    
    rag = MonolithicRAG(
        embed_model_name="BAAI/bge-small-en-v1.5",
        rerank_model_name="BAAI/bge-reranker-base",
        llm_model_name="Qwen/Qwen2.5-0.5B",
        db_config=db_config
    )
    
    await rag.init_db()
    
    passages = [
        "AI is a branch of computer science...",
        "Machine learning is a subset of AI...",
        "Deep learning revolutionized AI...",
        "OSDI is a conference about systems...",
        "ASPLOS is a conference about systems...",
        "MICRO is a conference about systems...",
        "HPCA is a conference about systems...",
        "MLSYS is a conference about systems..."
    ]
    queries = ["I want to know some system conferences."]
    
    results = await rag.run_pipeline(passages, queries)
    
    for res in results:
        print(f"\nFinal Result for '{res['query']}':")
        print(res['answer'])

if __name__ == "__main__":
    asyncio.run(main())
