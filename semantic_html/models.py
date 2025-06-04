from semantic_html.utils import generate_uuid
import re, json
from datetime import datetime, timezone

DEFAULT_CONTEXT={
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "doco": "http://purl.org/spar/doco/",
    "dcterms": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "@vocab": "https://semantic-html.org/vocab#",
    "Note": "doco:Document",
    "Structure": "doco:DiscourseElement",
    "Locator": "ex:Locator",
    "Doc": "doco:Section",
    "Annotation": "schema:Comment",
    "Quotation": "doco:BlockQuotation",
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
        "@id": "owl:sameAs",
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
    "generatedAtTime": {
        "@id": "prov:generatedAtTime",
        "@type": "xsd:dateTime"
    }
}

WADM_CONTEXT = "https://www.w3.org/ns/anno.jsonld"

class BaseGraphItem:
    """Base class for all graph items with standardized fields."""

    def __init__(self, type_, text=None, metadata=None, selector=None, **kwargs):


        self.data = {
            "@type": type_,
            "@id": generate_uuid(),            
            "generatedAtTime": datetime.now(timezone.utc).isoformat()
        }
        self.selector = selector  or {}
        self.data["text"] = text or ""
        self.wadm_metadata=kwargs.pop("wadm_meta", None).get("metadata", None)


        field_map = {
            "note_id": "note",
            "structure_id": "structure",
            "locator_id": "locator",
            "doc_id": "doc",
            "same_as": "sameAs",
            "html": "html"
        }

        for key, jsonld_key in field_map.items():
            if key in kwargs and kwargs[key] is not None:
                self.data[jsonld_key] = kwargs[key]

        if metadata:            
            self.data.update(metadata)

    def to_dict(self):
        """Return the graph item as a dictionary."""
        return self.data
    
    def to_wadm(self):
        """Return a WADM-conformant dictionary representation."""
        return generate_wadm_annotation(self)
    
class NoteItem(BaseGraphItem):
    def __init__(self, text, **kwargs):
        type_ = kwargs.pop("type_", ["Note"])
        super().__init__(type_=type_, text=text, **kwargs)

class StructureItem(BaseGraphItem):
    def __init__(self, text, level, **kwargs):
        type_ = kwargs.pop("type_", ["Structure"])
        super().__init__(type_=type_, text=text, **kwargs)
        self.data["level"] = level

class LocatorItem(BaseGraphItem):
    def __init__(self, text, **kwargs):
        type_ = kwargs.pop("type_", ["Locator"])
        super().__init__(type_=type_, text=text, **kwargs)

class DocItem(BaseGraphItem):
    def __init__(self, text, **kwargs):
        type_ = kwargs.pop("type_", ["Doc"])
        super().__init__(type_=type_, text=text, **kwargs)

class AnnotationItem(BaseGraphItem):
    def __init__(self, text, **kwargs):
        type_ = kwargs.pop("type_", ["Annotation"])
        super().__init__(type_=type_, text=text, **kwargs)

class QuotationItem(BaseGraphItem):
    def __init__(self, text, **kwargs):
        type_ = kwargs.pop("type_", ["Quotation"])
        super().__init__(type_=type_, text=text, **kwargs)


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
    

def generate_wadm_annotation(item):
    data = item.data
    selector = item.selector
    wadm = {        
        "@type": "Annotation",
        "@id": generate_uuid(),
        "created": datetime.now().isoformat(),
        "motivation": "identifying" if "doc" in data else "describing",
        "target": {
            "source": data.get("doc", data.get("note", data.get("@id","n/a"))),
            "selector": []
        },
        "body": []
    }

    wadm.update(item.wadm_metadata)

    selector_items = []

    # TextQuoteSelector
    if "text" in data and "suffix" in selector and "prefix" in selector:
        selector_items.append({
            "type": "TextQuoteSelector",
            "exact": data.get("text"),
            "prefix": selector.get("prefix"),
            "suffix": selector.get("suffix")
        })

    # TextPositionSelector
    if "start" in selector and "end" in selector:
        selector_items.append({
            "type": "TextPositionSelector",
            "start": selector.get("start"),
            "end": selector.get("end")
        })

    # XPathSelector
    if selector.get("tag"):
        selector_items.append({
            "type": "XPathSelector",
            "value": f"//{selector['tag']}"
        })

    # CssSelector
    if selector.get("style"):
        selector_items.append({
            "type": "CssSelector",
            "value": selector["style"]
        })

    if selector_items:
        wadm["target"]["selector"] = {
            "type": "Choice",
            "items": selector_items
        }

    scope = [
        data[key] for key in ["note", "structure", "locator"]
        if key in data
    ]

    if scope:
        wadm["target"]["scope"] = scope[0] if len(scope) == 1 else scope


    # body: identifying
    wadm["body"].append({
        "type": "SpecificResource",
        "source": data["@id"],
        "purpose": "identifying",
        "format": "text/plain"
    })

    # body: tagging
    if "@type" in data:
        types = data["@type"] if isinstance(data["@type"], list) else [data["@type"]]
        for t in types:
            wadm["body"].append({
                "type": "TextualBody",
                "value": t,
                "purpose": "tagging"
            })

    return wadm