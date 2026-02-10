from rapidfuzz import fuzz
from sqlalchemy import create_engine, text
from models import Base
import re
import yake
from collections import Counter

DB_PATH = "2025_11_09_researchgate.sqlite"
ENGINE_URL = f"sqlite:///{DB_PATH}"

STOPWORDS = {
    # standard english
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "had",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "without",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "while",
    "why",
    "how",
    "than",
    "then",
    "there",
    "here",
    "also",
    "only",
    "more",
    "most",
    "less",
    "few",
    "many",
    "some",
    "any",
    "all",
    "each",
    "both",
    "either",
    "neither",
    "not",
    "no",
    "nor",
    "so",
    "too",
    "very",
    "can",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "doing",
    "done",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "they",
    "them",
    "their",
    "theirs",
    "i",
    "me",
    "my",
    "mine",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    # abstract boilerplate / discourse
    "paper",
    "article",
    "study",
    "studies",
    "research",
    "work",
    "proposed",
    "propose",
    "introduce",
    "introduced",
    "present",
    "presented",
    "demonstrate",
    "demonstrated",
    "show",
    "shows",
    "shown",
    "evaluate",
    "evaluated",
    "evaluation",
    "experiment",
    "experiments",
    "result",
    "results",
    "analysis",
    "performance",
    "method",
    "methods",
    "approach",
    "approaches",
    "framework",
    "system",
    "systems",
    "model",
    "models",
    "algorithm",
    "algorithms",
    "technique",
    "techniques",
    "strategy",
    "strategies",
    "process",
    "processes",
    "based",
    "using",
    "use",
    "used",
    "utilize",
    "utilized",
    "including",
    "include",
    "included",
    "consist",
    "consists",
    "provide",
    "provides",
    "provided",
    "enable",
    "enables",
    "achieve",
    "achieved",
    "achieves",
    "improve",
    "improved",
    "improves",
    "enhance",
    "enhanced",
    "enhances",
    "significant",
    "significantly",
    "effective",
    "effectively",
    "efficient",
    "efficiency",
    "robust",
    "robustness",
    "novel",
    "new",
    "recent",
    "recently",
    "current",
    "future",
    "state",
    "state-of-the-art",
    # vague quantifiers / glue
    "various",
    "multiple",
    "different",
    "several",
    "numerous",
    "overall",
    "general",
    "main",
    "primary",
    "key",
    "first",
    "second",
    "third",
    "finally",
    "last",
    "one",
    "two",
    "three",
    "four",
    "five",
    # problem framing
    "problem",
    "problems",
    "issue",
    "issues",
    "challenge",
    "challenges",
    "limitation",
    "limitations",
    "difficulty",
    "difficult",
    "complex",
    "complexity",
    # reporting / comparison
    "compared",
    "comparison",
    "according",
    "therefore",
    "thus",
    "however",
    "moreover",
    "further",
    "furthermore",
    "additionally",
    "respectively",
    "namely",
    # junk / artifacts
    "et",
    "al",
    "via",
    "per",
    "https",
    "http",
    "www",
    "com",
    "org",
    "github",
}


def init_db():
    """Initialize the DB if needed."""
    engine = create_engine(ENGINE_URL, future=True)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)


def find_generic_terms(abstracts, df_threshold=0.05, min_token_len=2):
    n_docs = 0
    df = Counter()
    token_re = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

    for abstract_text in abstracts:
        n_docs += 1
        tokens = set(
            t
            for t in token_re.findall(clean_text(abstract_text).lower())
            if len(t) >= min_token_len
        )
        df.update(tokens)

    cutoff = df_threshold * n_docs
    generic = {t for t, c in df.items() if c >= cutoff}
    return generic, df, n_docs


def dedup_token_set(keywords, thresh):
    kept = []
    for kw, score in sorted(keywords, key=lambda x: x[1]):
        kw_len = len(kw.split())
        replace_idx = None
        drop = False
        for i, (k2, s2) in enumerate(kept):
            if fuzz.token_set_ratio(kw, k2) >= thresh:
                if kw_len < len(k2.split()):
                    replace_idx = i
                else:
                    drop = True
                break
        if replace_idx is not None:
            kept[replace_idx] = (kw, score)
        elif not drop:
            kept.append((kw, score))
    return kept


def extract_kw(raw_text, generic_terms):
    """Extract the keywoards out of the `raw_text` field."""
    kw_list = []
    for n_gram_size, top_n, win_size in [(3, 30, 2), (6, 50, 3)]:
        custom_kw_extractor = yake.KeywordExtractor(
            lan="en",  # language
            n=n_gram_size,  # ngram size
            dedupLim=0.9,  # deduplication threshold
            dedupFunc="seqm",  # deduplication function
            windowsSize=win_size,  # context window
            top=top_n,  # number of keywords to extract
            features=None,  # custom features
        )
        raw_keywords = custom_kw_extractor.extract_keywords(raw_text)
        kw_list.extend(raw_keywords)
    deduped_keywords = dedup_token_set(kw_list, thresh=90)
    out = set()
    for kw, score in deduped_keywords:
        toks = kw.lower().split()
        if all(token in generic_terms for token in toks) or any(
            token in STOPWORDS for token in toks
        ):
            continue
        out.add((kw, score))
    return out


def remove_stopwords(text, stopwords):
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return " ".join(t for t in tokens if t not in stopwords)


def clean_text(text):
    token_re = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
    return " ".join(token_re.findall(text.lower()))


def extract_abstract_batch(id_range):
    """Fetch abstracts for publications within an inclusive ID range.

    Args:
        id_range (tuple[int, int]): A (low, high) tuple specifying the inclusive
            range of publication IDs to fetch.

    Returns:
        list[str]: A list of abstract texts for publications whose IDs fall
            within the specified range. Publications with NULL abstracts
            are excluded.
    """
    low, high = id_range
    engine = create_engine(ENGINE_URL, future=True)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT abstract
                FROM publications
                WHERE id >= :low
                  AND id <= :high
                  AND abstract IS NOT NULL
                ORDER BY id
                """
            ),
            {"low": low, "high": high},
        ).all()

    return [r[0] for r in rows]


def main():
    """Entry point."""
    init_db()
    abstract_list = extract_abstract_batch((1, 10000))
    generic_terms, df, n_docs = find_generic_terms(
        abstract_list, df_threshold=0.15, min_token_len=2
    )
    print(generic_terms)
    for abstract_text in abstract_list:
        print(abstract_text)
        keywords = extract_kw(abstract_text, generic_terms)
        print(
            "\n".join(
                [
                    f"{x[0]} - {x[1]}"
                    for x in sorted(keywords, key=lambda x: -x[1])
                ]
            )
        )
        return


if __name__ == "__main__":
    main()
