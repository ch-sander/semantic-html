from semantic_html.utils import generate_uuid
import re
from datetime import datetime, timezone

DEFAULT_CONTEXT={

    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "doco": "http://purl.org/spar/doco/",
    "dcterms": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "@vocab": "https://semantic-html.org/vocab#",
    "Note": "doco:Document",
    "Structure": "doco:DiscourseElement",
    "Locator": "ex:Locator",
    "Doc": "doco:Section",
    "Annotation": "schema:Comment",
    "Quotation": "doco:BlockQuotation",
    "text": {
        "@id": "schema:text",
        "@type": "xsd:string"
    },
    "html": {
        "@id": "schema:html",
        "@type": "xsd:string"
    },
    "note": {
        "@id": "inNote",
        "@type": "@id"
    },
    "structure": {
        "@id": "inStructure",
        "@type": "@id"
    },
    "locator": {
        "@id": "hasLocator",
        "@type": "@id"
    },
    "sameAs": {
        "@id": "schema:sameAs",
        "@type": "@id"
    },
    "doc": {
        "@id": "dcterms:isPartOf",
        "@type": "@id"
    },
    "level": {
        "@id": "doco:hasLevel",
        "@type": "xsd:int"
    },
    "start": {
        "@id": "schema:startOffset",
        "@type": "xsd:int"
    },
    "end": {
        "@id": "schema:endOffset",
        "@type": "xsd:int"
    },
    "generatedAtTime": {
        "@id": "prov:generatedAtTime",
        "@type": "xsd:dateTime"
    }

}

class BaseGraphItem:
    """Base class for all graph items with standardized fields."""

    def __init__(self, type_, text=None, note_id=None,
                 structure_id=None, locator_id=None,
                 same_as=None, html=None, metadata=None):
        self.data = {
            "@type": type_,
            "@id": generate_uuid(),            
            "generatedAtTime": datetime.now(timezone.utc).isoformat()
        }
        if text is not None:
            self.data["text"] = text
        if html is not None:
            self.data["html"] = html
        if note_id is not None:
            self.data["note"] = note_id
        if structure_id is not None:
            self.data["structure"] = structure_id
        if locator_id is not None:
            self.data["locator"] = locator_id
        if same_as is not None:
            self.data["sameAs"] = same_as

        if metadata:
            for key, value in metadata.items():
                self.data[key] = value

    def to_dict(self):
        """Return the graph item as a dictionary."""
        return self.data

class NoteItem(BaseGraphItem):
    """Graph item representing a full note."""
    def __init__(self, text, type_=["Note"], html=None, note_id=None, metadata=None):
        super().__init__(type_=type_, text=text,
                         html=html, note_id=note_id,
                         metadata=metadata)

class StructureItem(BaseGraphItem):
    """Graph item representing a document structure element (e.g., heading)."""
    def __init__(self, text, level, note_id,
                 type_=["Structure"], structure_id=None, locator_id=None):
        super().__init__(type_=type_, text=text,
                         structure_id=structure_id,
                         locator_id=locator_id,
                         note_id=note_id)
        self.data["level"] = level

class LocatorItem(BaseGraphItem):
    """Graph item representing a locator (e.g., page reference)."""
    def __init__(self, text, structure_id, note_id,
                 type_=["Locator"]):
        super().__init__(type_=type_, text=text,
                         structure_id=structure_id,
                         note_id=note_id)

class DocItem(BaseGraphItem):
    """Graph item representing a document text block."""
    def __init__(self, text, structure_id, locator_id, note_id,
                 type_=["Doc"]):
        super().__init__(type_=type_, text=text,
                         structure_id=structure_id,
                         locator_id=locator_id,
                         note_id=note_id)

class AnnotationItem(BaseGraphItem):
    """Graph item representing an annotation."""
    def __init__(self, text, start, end, doc_id,
                 structure_id=None, locator_id=None, note_id=None,
                 same_as=None, type_=["Annotation"]):
        super().__init__(type_=type_, text=text,
                         structure_id=structure_id,
                         locator_id=locator_id,
                         note_id=note_id,
                         same_as=same_as)
        if start >= 0:
            # Store offsets as raw ints; context defines xsd:int
            self.data["start"] = start
            self.data["end"] = end
            self.data["doc"] = doc_id

class QuotationItem(BaseGraphItem):
    """Graph item representing a quotation block."""
    def __init__(self, text, structure_id, locator_id, note_id,
                 type_=["Quotation"]):
        super().__init__(type_=type_, text=text,
                         structure_id=structure_id,
                         locator_id=locator_id,
                         note_id=note_id)

class RegexWrapper:
    def __init__(self, mapping: dict):
        self.regex_entries = self._extract_regex_entries(mapping)

    def _extract_regex_entries(self, mapping: dict) -> list:
        entries = []

        def recurse(submapping: dict):
            for key, value in submapping.items():
                if key == "IGNORE":
                    continue 
                if isinstance(value, dict):
                    regex_list = value.get("regex")
                    if regex_list:
                        class_name = value.get("class", key)
                        value.setdefault("class", class_name)
                        tags = value.setdefault("tags", [])
                        if "span" not in tags:
                            tags.append("span")

                        if isinstance(regex_list, str):
                            regex_list = [regex_list]

                        for pattern in regex_list:
                            entries.append((class_name, pattern, value.get("types", [])))

                    recurse(value)

        recurse(mapping)
        return entries

    def wrap(self, html: str) -> str:
        if not self.regex_entries:
            return html

        for cls, pattern, _ in self.regex_entries:
            html = self._wrap_pattern(html, pattern, cls)
        return html

    def _wrap_pattern(self, html: str, pattern: str, cls: str) -> str:
        def replacer(match):
            return f'<span class="{cls}">{match.group(0)}</span>'
        return re.sub(pattern, replacer, html)

