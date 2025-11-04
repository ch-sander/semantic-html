import spacy
from spacy.training import Example
from pathlib import Path
from lxml import etree, html

def extract_plaintext_with_map(html_string):
    """
    Extract plain text from HTML and build a mapping of offsets → (element, text, start, end).
    """
    tree = html.fromstring(html_string)
    plain = []
    offset_map = []
    offset = 0

    for node in tree.xpath("//text()"):
        text = str(node)
        if not text.strip():
            continue
        start = offset
        end = offset + len(text)
        offset_map.append((start, end, node.getparent(), text))
        plain.append(text)
        offset = end + 1
    return " ".join(plain), offset_map, tree

def split_node_with_entities(text, rel_entities, label2tag):
    """
    Replace the text of a node with multiple child nodes, inserting entity tags.
    Leaves normal text as plain text, no extra <span>.
    """
    result_nodes = []
    rel_entities = sorted(rel_entities, key=lambda x: x["start"])
    cursor = 0

    for ent in rel_entities:
        s, e, label = ent["start"], ent["end"], ent["label"]
        if s > cursor:
            result_nodes.append(text[cursor:s])
        tag = label2tag.get(label, "span")
        el = etree.Element(tag)
        el.text = text[s:e]
        result_nodes.append(el)
        cursor = e

    if cursor < len(text):
        result_nodes.append(text[cursor:])

    return result_nodes


def entities_to_markdown(text, entities, label2md):
    """
    Wrap entities in a plain text string with Markdown based on label2md mapping.
    entities: list of {"start": int, "end": int, "label": str, "text": str}
    label2md: dict like {"PERSON": ("*", "*"), "LOC": ("`", "`")}
    """
    # sort by start offset
    entities = sorted(entities, key=lambda x: x["start"])
    md_parts = []
    cursor = 0

    for ent in entities:
        s, e, label = ent["start"], ent["end"], ent["label"]
        if s > cursor:
            md_parts.append(text[cursor:s])

        prefix, suffix = label2md.get(label, ("", ""))
        md_parts.append(f"{prefix}{text[s:e]}{suffix}")
        cursor = e

    if cursor < len(text):
        md_parts.append(text[cursor:])

    return "".join(md_parts)

def project_entities_to_html(html_string, entities, label2tag):
    plain, offset_map, tree = extract_plaintext_with_map(html_string)

    parent_entities = {}
    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        for (s, e, parent, text) in offset_map:
            if s <= start < e:
                rel_start = start - s
                rel_end = min(end, e) - s
                parent_entities.setdefault((parent, text), []).append(
                    {"start": rel_start, "end": rel_end, "label": label}
                )
                break

    for (parent, text), rel_entities in parent_entities.items():
        # Clear text in parent
        if parent.text == text:
            parent.text = None
        else:
            for child in parent:
                if child.tail == text:
                    child.tail = None

        # Insert segments back
        for part in split_node_with_entities(text, rel_entities, label2tag):
            if isinstance(part, str):
                if len(parent) == 0 and parent.text is None:
                    parent.text = part
                else:
                    if parent[-1].tail is None:
                        parent[-1].tail = part
                    else:
                        parent[-1].tail += part
            else:
                parent.append(part)

    return etree.tostring(tree, pretty_print=True, encoding="unicode")

class SpacyNERWrapper:
    def __init__(self, labels=None, model_dir="ner_model", lang="en"):
        """
        labels: optional list of entity labels (e.g. ["PERSON", "LOC"])
        model_dir: directory where the model will be saved/loaded
        """
        self.labels = labels or []
        self.lang = lang
        self.model_dir = Path(model_dir)
        self.nlp = None

    # ---------------- TRAINING ----------------
    def train(self, conll_data, n_iter=20, resume=False):
        """
        Train a spaCy NER model on CoNLL-style data.
        conll_data: string in CoNLL format (token \t label)
        """
        # Parse CoNLL
        sentences = []
        sent_tokens, sent_labels = [], []
        for line in conll_data.splitlines():
            if not line.strip():
                if sent_tokens:
                    sentences.append((sent_tokens, sent_labels))
                    sent_tokens, sent_labels = [], []
                continue
            tok, lab = line.split()
            sent_tokens.append(tok)
            sent_labels.append(lab)
        if sent_tokens:
            sentences.append((sent_tokens, sent_labels))
        if not self.labels:

            self.labels = list({lab[2:] for _, labs in sentences for lab in labs if lab != "O"})

        if resume and self.model_dir.exists():
            print("Resuming training from existing model...")
            self.nlp = spacy.load(self.model_dir)
            ner = self.nlp.get_pipe("ner")
        else:
            print("Training from scratch...")
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe("ner")
            else:
                ner = self.nlp.get_pipe("ner")

        # Add labels

        for lab in self.labels:
            ner.add_label(lab)

        # Prepare training data
        train_data = []
        for tokens, labels in sentences:
            text = " ".join(tokens)
            ents = []
            offset = 0
            for tok, lab in zip(tokens, labels):
                start = text.find(tok, offset)
                end = start + len(tok)
                if lab.startswith("B-"):
                    ents.append([start, end, lab[2:]])
                elif lab.startswith("I-") and ents:
                    ents[-1][1] = end
                offset = end
            train_data.append((text, {"entities": ents}))

        optimizer = self.nlp.initialize() if not resume else self.nlp.resume_training()
        for itn in range(n_iter):
            losses = {}
            examples = [Example.from_dict(self.nlp.make_doc(text), ann) for text, ann in train_data]
            self.nlp.update(examples, sgd=optimizer, losses=losses)
            print(f"Iteration {itn+1}, Losses: {losses}")

        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.nlp.to_disk(self.model_dir)
        print(f"Model saved to {self.model_dir}")

    # ---------------- PREDICTION ----------------
    def load(self):
        """Load model from disk."""
        if self.model_dir.exists():
            self.nlp = spacy.load(self.model_dir)
        else:
            raise ValueError(f"No model found in {self.model_dir}")

    def predict_html(self, html_string, label2tag):
        """
        Run NER on HTML:
        - extract plain text and offset map
        - predict entities
        - re-inject into HTML
        """
        if self.nlp is None:
            self.load()

        plain, offset_map, tree = extract_plaintext_with_map(html_string)
        doc = self.nlp(plain)
        entities = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text}
                    for ent in doc.ents]

        return project_entities_to_html(html_string, entities, label2tag)
    
    def predict_markdown(self, plain_text, label2md):
        if self.nlp is None:
            self.load()

        doc = self.nlp(plain_text)
        entities = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text}
                    for ent in doc.ents]

        return entities_to_markdown(plain_text, entities, label2md)