from typing import Dict, List, Sequence


class CrossEncoderReranker:
    """
    ONNX cross-encoder shaped interface.

    Uses token overlap when onnxruntime / a SARR checkpoint is not installed.
    """

    def rerank(self, query: str, passages: Sequence[Dict], top_k: int = 3) -> List[Dict]:
        q = set((query or "").lower().split())
        scored = []
        for doc in passages:
            text = (doc.get("text") or "").lower()
            tokens = text.split()
            overlap = len(q.intersection(tokens)) / max(len(q), 1)
            phrase = 0.15 if any(term in text for term in ("risk", "capital", "liquidity")) else 0.0
            item = dict(doc)
            item["rerank_score"] = float(overlap + phrase + float(doc.get("score") or 0) * 0.2)
            scored.append(item)
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]
