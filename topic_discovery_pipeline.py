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


_token_re = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def _is_acronym_or_model(tok):
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
    return len(toks) <= max_len and toks[-1] in TRUNC_HEADS


def extract_kw(raw_text, generic_terms, hyper_params):
    """Extract the keywoards out of the `raw_text` field."""
    kw_list = []
    for n_gram_size, top_n, win_size in hyper_params:
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
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return " ".join(t for t in tokens if t not in stopwords)


def clean_text(text):
    token_re = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
    return " ".join(token_re.findall(text.lower()))


def extract_publications_with_abstract(id_range):
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
