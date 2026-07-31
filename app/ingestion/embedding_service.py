from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from utils.logger_config import logger

bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
def build_embeddings(text: str, embedding_model: BGEM3FlagModel) -> tuple[list[float], dict[str, list[float]]]:
    """Generate dense and sparse embeddings using bgem3 embedding model and fastembed"""
    embeddings = embedding_model.encode(
        text,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    dense_embedding = embeddings["dense_vecs"]
    bm25_result = list(bm25_model.embed([text]))[0]
    sparse_embedding = {
        "indices": bm25_result.indices.tolist(),
        "values": bm25_result.values.tolist()
    }
    return dense_embedding, sparse_embedding
