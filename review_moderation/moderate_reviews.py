"""Review moderation.

Flags customer reviews that shouldn't be published: profanity, links/spam,
or all-caps rants. Clean reviews go to reviews.approved.

Input:  reviews.raw (review_id, sku, rating, text)
Output: reviews.approved, reviews.flagged (review_id, reason)
"""
import re

BLOCKLIST = {"scam", "garbage", "idiot"}


def normalize(text):
    text = re.sub(r"http\S+", " <url> ", text.lower())
    return re.sub(r"[^a-z<>\s]", " ", text)


def moderate(review):
    raw = review["text"]
    text = normalize(raw)
    if "<url>" in text:
        return "spam_link"
    if any(w in BLOCKLIST for w in text.split()):
        return "profanity"
    letters = [c for c in raw if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return "shouting"
    return None


def run(reviews):
    approved, flagged = [], []
    for r in reviews:
        reason = moderate(r)
        (flagged.append((r["review_id"], reason)) if reason else approved.append(r))
    return approved, flagged


if __name__ == "__main__":
    reviews = [
        {"review_id": 1, "sku": "BED-001", "rating": 5, "text": "Lovely duvet"},
        {"review_id": 2, "sku": "BED-001", "rating": 1, "text": "SCAM buy at http://cheap.example"},
    ]
    print(run(reviews))
