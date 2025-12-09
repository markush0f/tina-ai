from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.cli.download import download
import yake


class TopicService:
    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("es_core_news_md")
        except OSError:
            model = "es_core_news_md"
            # self.logger(f"Downloaded model {model}")
            download(model)
            self.nlp = spacy.load("es_core_news_md")

        # Load embedding model (change implemented here)
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_noun_phrases(self, text: str):
        """Extracts noun phrases and text structure"""
        doc = self.nlp(text)
        return [chunk.text.lower().strip() for chunk in doc.noun_chunks]

    def extract_keywords(self, text: str, max_kw=10):
        """Extract important keywords"""
        kw_extractor = yake.KeywordExtractor(lan="es", top=max_kw)
        keywords = kw_extractor.extract_keywords(text)
        return [kw for kw, score in keywords]

    def normalize_topic(self, topic: str):
        """
        Normalize a topic string by converting it into a clean and meaningful
        semantic representation.

        This method processes a raw topic candidate (often noisy or unstructured),
        and extracts only the essential semantic components by applying:

        - Lemmatization: reduces words to their base form
        (e.g., "relajé" -> "relajación", "ansioso" -> "ansioso").

        - POS filtering: keeps only nouns and adjectives, which are the most
        informative parts of a topic. Verb-only phrases or irrelevant terms
        are discarded entirely (e.g., "sentí" -> "").

        - Lowercasing and whitespace cleanup.

        The goal is to convert raw phrases into clean concept-level representations
        that can be consistently compared using embeddings inside the deduplication step.

        Example:
            Input:  "gimnasio me relajé"
            Output: "gimnasio relajación"

            Input:  "sentí muy ansioso"
            Output: "ansioso"

            Input:  "relajé"
            Output: ""

        Returns:
            str: A normalized topic string containing only meaningful semantic units.
        """

        doc = self.nlp(topic)

        # Keep only lemmas from nouns and adjectives
        tokens = [
            token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"]
        ]

        return " ".join(tokens).strip()

    def dedupe_topics(self, topics: list[str], threshold: float = 0.75) -> list[str]:
        """Remove semantically duplicate topics using embeddings"""
        if not topics:
            return []

        # Convert topics to vectors and generate a tensor
        embeddings = self.emb_model.encode(topics, convert_to_tensor=True)
        final_topics = []

        for i, topic in enumerate(topics):
            is_duplicate = False

            for j in range(i):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()

                if sim > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_topics.append(topic)

        return final_topics

    def extract_candidates(self, text: str):
        """Combine keywords and noun phrases into a single list of candidates (bug fix implemented here)."""
        noun_phrases = self.extract_noun_phrases(text)
        keywords = self.extract_keywords(text)
        return list(set(noun_phrases + keywords))  # changed: replaced 'and' with '+'

    def extract_topics(self, text: str):
        """Full pipeline: extract, normalize, and dedupe topics (change implemented here)."""
        candidates = self.extract_candidates(text)

        # Normalize all candidates
        normalized = [self.normalize_topic(t) for t in candidates]
        normalized = [t for t in normalized if len(t) > 0]  # remove empty topics

        # Remove semantic duplicates
        final_topics = self.dedupe_topics(normalized)

        return final_topics


# TESTING
topic = TopicService()
text = "Hoy me sentí muy ansioso en clase pero luego en el gimnasio me relajé."
print(topic.extract_noun_phrases(text))
print(topic.extract_keywords(text))
candidates = topic.extract_candidates(text)
dedupe = topic.dedupe_topics(candidates)
print("DEDUPE: ", dedupe)
topics = topic.extract_topics(text)
print("Topics: ", topics)
# def show_model_location():
#     path = get_package_path("es_core_news_md")
#     print(f"Model installed at: {path}")


# show_model_location()
