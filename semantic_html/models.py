from semantic_html.utils import generate_uuid
import re, json
from lxml import etree
from datetime import datetime, timezone
from collections import defaultdict

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
        wadm_meta = kwargs.pop("wadm_meta", None)
        self.wadm_metadata = wadm_meta.get("metadata") if wadm_meta else None

        order_index = kwargs.pop("order_index", None)   # sequential per note
        if order_index is not None:
            self.data["orderIndex"] = int(order_index)

        tree_index  = kwargs.pop("tree_index", None)
        if tree_index is not None:
            self.data["treeIndex"] = int(tree_index)

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

    if item.wadm_metadata: wadm.update(item.wadm_metadata)

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
    if selector.get("xpath"):
        selector_items.append({
            "type": "XPathSelector",
            "value": selector['xpath']
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
        "purpose": "identifying"
    })

    # body: tagging
    if "@type" in data:
        types = data["@type"] if isinstance(data["@type"], list) else [data["@type"]]
        for t in types:
            wadm["body"].append({
                "type": "TextualBody",
                "value": t,
                "purpose": "tagging",
                "format": "text/plain"
            })

    return wadm

def build_tei_from_items(base_items: list[BaseGraphItem]):
    NS = "http://www.tei-c.org/ns/1.0"
    XML = "http://www.w3.org/XML/1998/namespace"

    tei_root = etree.Element(f"{{{NS}}}TEI", nsmap={'tei': NS, 'xml': XML})
    etree.SubElement(tei_root, f"{{{NS}}}teiHeader")
    body_el = etree.SubElement(etree.SubElement(tei_root, f"{{{NS}}}text"), f"{{{NS}}}body")

    structure_map, div_map, parent_links = {}, {}, {}
    doc_map, annotations_by_doc = {}, defaultdict(list)
    pending_locators, pending_quotations = [], []
    locator_docs = set()

    for item in base_items:
        item_id = item.data['@id']
        if isinstance(item, StructureItem):
            structure_map[item_id] = item
            parent_links[item_id] = item.data.get('structure')

        elif isinstance(item, LocatorItem):
            pending_locators.append(item)
            locator_docs.add(item.data.get('doc'))

        elif isinstance(item, QuotationItem):
            pending_quotations.append(item)

        elif isinstance(item, DocItem):
            doc_map[item_id] = item

        elif isinstance(item, AnnotationItem):
            doc_id = item.data.get('doc') or item.data.get('structure')
            if doc_id:
                annotations_by_doc[doc_id].append(item)

    def build_div(sid):
        if sid in div_map:
            return div_map[sid]
        item = structure_map[sid]
        div = etree.Element(f"{{{NS}}}div")
        div.set(f"{{{XML}}}id", sid)
        head = etree.SubElement(div, f"{{{NS}}}head")
        head.set(f"{{{XML}}}id", sid)
        head.text = item.data.get("text", "")
        div_map[sid] = div
        parent_id = parent_links.get(sid)
        if parent_id and parent_id not in div_map:
            build_div(parent_id)
        parent = div_map.get(parent_id, body_el)
        parent.append(div)
        return div

    for sid in structure_map:
        build_div(sid)

    for item in pending_locators:
        sid = item.data.get('structure')
        div = div_map.get(sid, body_el)
        milestone = etree.Element(f"{{{NS}}}milestone", unit="page", n=item.data.get("text", ""))
        milestone.set(f"{{{XML}}}id", item.data['@id'])
        div.append(milestone)

    for item in pending_quotations:
        sid = item.data.get('structure')
        div = div_map.get(sid, body_el)
        quote = etree.Element(f"{{{NS}}}quote")
        quote.set(f"{{{XML}}}id", item.data['@id'])
        quote.text = item.data.get("text", "")
        div.append(quote)

    for doc_id, doc_item in doc_map.items():
        full_text = doc_item.data.get("text", "")
        if not full_text:
            continue
        el_p = etree.Element(f"{{{NS}}}p")
        el_p.set(f"{{{XML}}}id", doc_id)

        inserts = [
            (ann.selector['start'], ann.selector['end'], "name", ann.data["@id"], full_text[ann.selector['start']:ann.selector['end']], ann.data.get("sameAs"))
            for ann in annotations_by_doc.get(doc_id, [])
        ]
        inserts.sort(reverse=True, key=lambda x: x[0])

        chunks = []
        last = len(full_text)
        for start, end, tag, aid, frag, target in inserts:
            if end > last:
                continue
            if full_text[end:last]:
                chunks.insert(0, full_text[end:last])
            el = etree.Element(f"{{{NS}}}{tag}")
            el.set(f"{{{XML}}}id", aid)
            if target:
                el.set("target", target)
            el.text = frag
            chunks.insert(0, el)
            last = start
        if last > 0:
            chunks.insert(0, full_text[:last])

        if isinstance(chunks[0], str):
            el_p.text = chunks[0]
            chunks = chunks[1:]
        for chunk in chunks:
            if isinstance(chunk, str):
                if len(el_p):
                    el_p[-1].tail = (el_p[-1].tail or '') + chunk
                else:
                    el_p.text = (el_p.text or '') + chunk
            else:
                el_p.append(chunk)

        div = div_map.get(doc_item.data.get('structure'), body_el)
        div.append(el_p)

    return tei_root
