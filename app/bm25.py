"""Small, dependency-free BM25 (Okapi) implementation for keyword scoring."""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "what", "which", "who", "whom", "when", "where", "how", "why", "of", "to", "in",
    "on", "at", "for", "and", "or", "it", "its", "this", "that", "with", "as", "by",
    "about", "any", "has", "have", "had", "tell", "me", "my", "i", "you", "your",
    "user", "users",
}


def stem(token: str) -> str:
    """Very light suffix stripping so 'uses'/'use', 'building'/'build' match."""
    for suffix in ("ing", "es", "s", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9\+#\.]+", text.lower().replace("'s", ""))
    tokens = [t.strip(".") for t in tokens]
    return [stem(t) for t in tokens if t and t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        """Raw BM25 score per document, plus normalisation to [0,1] helper below."""
        q = tokenize(query)
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d) or 1
            for t in q:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out

    def normalised(self, query: str) -> list[float]:
        """Score divided by the best score a document could get for this query."""
        q = set(tokenize(query))
        ceiling = sum(self.idf.get(t, 0.0) * (self.k1 + 1) / (1 + self.k1 * (1 - self.b)) for t in q)
        if ceiling <= 0:
            return [0.0] * self.n
        return [min(1.0, s / ceiling) for s in self.scores(query)]
