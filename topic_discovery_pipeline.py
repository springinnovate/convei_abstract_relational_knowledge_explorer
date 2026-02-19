"""
Extract YAKE keywords from publication abstracts in a SQLite database and store
deduplicated topics with a many-to-many mapping back to publications.

The script:
- Initializes the SQLAlchemy schema.
- Loads publications with non-null abstracts over an ID range.
- Computes corpus-level "generic terms" by document frequency threshold.
- Extracts candidate keywords per abstract via YAKE across multiple hyperparameter
  settings, then deduplicates and filters keywords using heuristics.
- Upserts topics into RawTopics and links them to publications via
  RawTopicToPublication.

Database:
  Uses a SQLite file at DB_PATH and connects via ENGINE_URL.

Filtering heuristics:
  Removes phrases that start with BAD_START, end with BAD_VERB, contain STOPWORDS,
  are composed entirely of corpus-generic terms, or look like truncated head phrases.
"""

from tqdm import tqdm
from rapidfuzz import fuzz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base
import re
import yake
from collections import Counter
from models import Publication, RawTopics, RawTopicToPublication

DB_PATH = "2025_11_09_researchgate.sqlite"
ENGINE_URL = f"sqlite:///{DB_PATH}"

BAD_START = {
    "examining",
    "using",
    "learning",
    "capturing",
    "improving",
    "improved",
    "improve",
    "addressing",
    "address",
    "extracting",
    "extract",
    "based",
    "show",
    "shows",
    "shown",
    "found",
    "finding",
    "demonstrate",
    "demonstrated",
    "providing",
    "provide",
    "proposes",
}

BAD_VERB = {
    "necessitates",
    "require",
    "requires",
    "causes",
    "cause",
    "leads",
    "lead",
    "affect",
    "affects",
    "enable",
    "enables",
    "improve",
    "improves",
    "increase",
    "increases",
    "reduce",
    "reduces",
    "make",
    "makes",
}

TRUNC_HEADS = {
    "remote",
    "sensing",
    "image",
    "images",
    "picture",
    "pictures",
    "technology",
}

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
    """Initializes the database schema and enables SQLite foreign keys.

    Creates a SQLAlchemy engine for the configured SQLite database, enables
    foreign key enforcement (SQLite PRAGMA), and creates all tables declared
    on the SQLAlchemy declarative Base metadata.

    Returns:
        None
    """
    engine = create_engine(ENGINE_URL, future=True)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)


def find_generic_terms(abstracts, df_threshold=0.05, min_token_len=2):
    """Finds corpus-generic tokens by document frequency.

    Tokenizes each abstract into a set of unique normalized tokens, counts in
    how many documents each token appears (document frequency), and marks tokens
    as "generic" if they appear in at least `df_threshold * n_docs` documents.

    Tokenization uses lowercase alphanumeric tokens with optional internal
    hyphens (e.g., 'state-of-the-art') and applies a minimum token length.

    Args:
        abstracts (Iterable[str]): Abstract texts to analyze.
        df_threshold (float): Minimum fraction of documents a token must appear
            in to be considered generic.
        min_token_len (int): Minimum token length to include.

    Returns:
        tuple[set[str], collections.Counter, int]:
            - generic: Tokens with document frequency >= cutoff.
            - df: Counter mapping token -> document frequency.
            - n_docs: Number of documents processed.
    """
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


_token_re = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def _is_acronym_or_model(tok):
    """Heuristically detects acronyms or model identifiers.

    Treats tokens as acronyms/models when they:
    - Contain any digit (e.g., 'BERT2', 'ResNet50'),
    - Mix lowercase and uppercase (e.g., 'iPhone', 'eBay'),
    - Have >= 2 alphabetic characters and all alphabetic characters are uppercase
      (e.g., 'CNN', 'LSTM').

    Args:
        tok (str): Single token to classify.

    Returns:
        bool: True if the token is likely an acronym or model identifier.
    """
    if any(ch.isdigit() for ch in tok):
        return True
    if any(ch.islower() for ch in tok) and any(ch.isupper() for ch in tok):
        return True
    letters = [ch for ch in tok if ch.isalpha()]
    if len(letters) >= 2 and all(ch.isupper() for ch in letters):
        return True
    return False


def dedup_token_set(
    keywords,
    thresh,
    min_tokens=2,
    keep_singletons=True,
):
    """Deduplicates keyword phrases using token-set fuzzy similarity.

    Iterates over candidate keywords, normalizes each to tokens, and drops or
    replaces near-duplicates based on `rapidfuzz.fuzz.token_set_ratio`.
    When two phrases are similar above `thresh`, the shorter tokenized phrase is
    preferred. Single-token phrases are optionally allowed only when they look
    like acronyms/models.

    Args:
        keywords (Iterable[tuple[str, float]]): Pairs of (keyword, score) where
            lower scores are assumed to be better (as produced by YAKE).
        thresh (int): Similarity threshold (0-100) for considering two phrases
            duplicates.
        min_tokens (int): Minimum number of tokens required for multi-token
            phrases. Single-token phrases are handled by `keep_singletons`.
        keep_singletons (bool): Whether to keep single-token keywords if they
            are classified as acronyms/models.

    Returns:
        list[tuple[str, float]]: Deduplicated (keyword, score) pairs.
    """
    kept = []
    for kw, score in sorted(keywords, key=lambda x: x[1]):
        toks = _token_re.findall(kw)
        if len(toks) == 1 and min_tokens > 1:
            if not (keep_singletons and _is_acronym_or_model(toks[0])):
                continue
        if len(toks) > 1 and len(toks) < min_tokens:
            continue

        kw_len = len(toks)
        replace_idx = None
        drop = False
        for i, (k2, s2) in enumerate(kept):
            if fuzz.token_set_ratio(kw, k2) >= thresh:
                if kw_len < len(_token_re.findall(k2)):
                    replace_idx = i
                else:
                    drop = True
                break

        if replace_idx is not None:
            kept[replace_idx] = (kw, score)
        elif not drop:
            kept.append((kw, score))
    return kept


def is_truncated_head_phrase(toks, max_len=3):
    """Checks whether a token list looks like a truncated head phrase.

    A phrase is considered truncated if it is short (<= `max_len`) and ends with
    a high-level head word that often appears as an incomplete fragment.

    Args:
        toks (Sequence[str]): Tokenized, normalized phrase.
        max_len (int): Maximum length for a phrase to be considered truncated.

    Returns:
        bool: True if the phrase matches the truncated-head heuristic.
    """
    return len(toks) <= max_len and toks[-1] in TRUNC_HEADS


def extract_kw(raw_text, generic_terms, hyper_params):
    """Extracts filtered keyword candidates from raw text using YAKE.

    Runs YAKE keyword extraction multiple times using `hyper_params`, merges all
    extracted candidates, deduplicates them, and filters them using heuristic
    rules:
    - Drop phrases that start with BAD_START or end with BAD_VERB.
    - Drop phrases consisting entirely of corpus-generic terms.
    - Drop phrases containing any STOPWORDS.
    - Drop truncated head phrases (e.g., short phrases ending in TRUNC_HEADS).

    Args:
        raw_text (str): Source text (typically an abstract).
        generic_terms (set[str]): Tokens considered corpus-generic.
        hyper_params (Iterable[tuple[int, int, int]]): YAKE parameter tuples of
            (n_gram_size, top_n, windows_size).

    Returns:
        set[tuple[str, float]]: Unique (keyword, score) pairs that pass filters.
            Scores are YAKE scores (lower is typically better).
    """
    kw_list = []
    for n_gram_size, top_n, win_size in hyper_params:
        custom_kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=n_gram_size,
            dedupLim=0.9,
            dedupFunc="seqm",
            windowsSize=win_size,
            top=top_n,
            features=None,
        )
        raw_keywords = custom_kw_extractor.extract_keywords(raw_text)
        kw_list.extend(raw_keywords)
    deduped_keywords = dedup_token_set(kw_list, thresh=90)
    out = set()
    for kw, score in deduped_keywords:
        toks = kw.lower().split()
        if (
            toks[0] in BAD_START
            or toks[-1] in BAD_VERB
            or all(token in generic_terms for token in toks)
            or any(token in STOPWORDS for token in toks)
        ):
            continue
        toks = [t.lower() for t in _token_re.findall(kw)]
        if is_truncated_head_phrase(toks):
            continue
        out.add((kw, score))
    return out


def remove_stopwords(text, stopwords):
    """Removes stopwords from text and returns a normalized token string.

    Tokenizes `text` into lowercase alphanumeric/hyphen tokens and removes any
    token present in `stopwords`.

    Args:
        text (str): Input text.
        stopwords (set[str]): Tokens to remove.

    Returns:
        str: Space-joined tokens with stopwords removed.
    """
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return " ".join(t for t in tokens if t not in stopwords)


def clean_text(text):
    """Normalizes text to lowercase alphanumeric/hyphen tokens.

    Extracts tokens matching `[a-z0-9]+(?:-[a-z0-9]+)*` from the lowercased text
    and returns them as a space-joined string.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized, space-joined tokens.
    """
    token_re = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
    return " ".join(token_re.findall(text.lower()))


def extract_publications_with_abstract(id_range):
    """Loads publication IDs and abstracts for publications with abstracts.

    Queries the Publication table for rows with IDs within `id_range` (inclusive)
    and a non-null abstract, ordered by ID.

    Args:
        id_range (tuple[int, int]): Inclusive (low, high) ID bounds.

    Returns:
        list[tuple[int, str]]: List of (publication_id, abstract_text).
    """
    low, high = id_range
    engine = create_engine(ENGINE_URL, future=True)
    Session = sessionmaker(bind=engine, future=True)

    with Session() as session:
        pubs = (
            session.query(Publication)
            .filter(Publication.id >= low, Publication.id <= high)
            .filter(Publication.abstract.isnot(None))
            .order_by(Publication.id)
            .all()
        )
        return [(p.id, p.abstract) for p in pubs]


def main():
    """Runs the end-to-end topic extraction and persistence pipeline.

    Initializes the schema, loads all publications with abstracts, computes
    corpus-level generic terms, extracts filtered YAKE topics per publication,
    and upserts topics and publication-topic links into the database.

    Returns:
        None
    """
    init_db()
    pubs = extract_publications_with_abstract((1, 99999999999))
    abstract_list = [a for _, a in pubs]

    generic_terms, df, n_docs = find_generic_terms(
        abstract_list, df_threshold=0.15, min_token_len=2
    )

    engine = create_engine(ENGINE_URL, future=True)
    Session = sessionmaker(bind=engine, future=True)

    hyper_params = [(3, 30, 2), (6, 50, 5)]

    topic_cache = {}
    with Session() as session:
        for t_id, t_topic in session.query(RawTopics.id, RawTopics.topic).all():
            topic_cache[t_topic] = t_id

    with Session() as session:
        session.execute(text("PRAGMA foreign_keys=ON"))

        for pub_id, abstract_text in tqdm(pubs):
            keywords = extract_kw(abstract_text, generic_terms, hyper_params)
            topics = [kw.strip() for kw, _ in keywords if kw and kw.strip()]
            if not topics:
                continue

            for topic in topics:
                topic_id = topic_cache.get(topic)
                if topic_id is None:
                    rt = RawTopics(topic=topic)
                    session.add(rt)
                    session.flush()
                    topic_id = rt.id
                    topic_cache[topic] = topic_id

                session.merge(
                    RawTopicToPublication(
                        topic_id=topic_id,
                        publication_id=pub_id,
                    )
                )

            session.commit()


if __name__ == "__main__":
    main()
