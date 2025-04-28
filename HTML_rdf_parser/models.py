from HTML_rdf_parser.utils import generate_uuid

class BaseGraphItem:
    """Base class for all graph items with standardized fields."""

    def __init__(self, type_, text=None, note_id=None, structure_id=None, locator_id=None, same_as=None):
        self.data = {
            "@id": generate_uuid(),
            "@type": type_,
        }
        if text is not None:
            self.data["text"] =  {"@value": text.strip(), "@type":"xsd:string"}
        if note_id is not None:
            self.data["note"] = {"@id": note_id}
        if structure_id is not None:
            self.data["structure"] = {"@id": structure_id}
        if locator_id is not None:
            self.data["locator"] = {"@id": locator_id}
        if same_as is not None:
            self.data["same:as"] = {"@id": same_as}

    def to_dict(self):
        """Return the graph item as a dictionary."""
        return self.data

class NoteItem(BaseGraphItem):
    """Graph item representing a full note."""
    def __init__(self, text, type_=["Note"], **kwargs):
        super().__init__(type_=type_, text=text, **kwargs)

class StructureItem(BaseGraphItem):
    """Graph item representing a document structure element (e.g., heading)."""
    def __init__(self, text, level, note_id, type_=["Structure"], **kwargs):
        super().__init__(type_=type_, text=text, note_id=note_id, **kwargs)
        self.data["level"] = {"@value": level, "@type":"xsd:int"}

class LocatorItem(BaseGraphItem):
    """Graph item representing a locator (e.g., page reference)."""
    def __init__(self, text, structure_id, note_id, type_=["Locator"], **kwargs):
        super().__init__(type_=type_, text=text, structure_id=structure_id, note_id=note_id, **kwargs)

class DocItem(BaseGraphItem):
    """Graph item representing a document text block."""
    def __init__(self, text, structure_id, locator_id, note_id, type_=["Doc"], **kwargs):
        super().__init__(type_=type_, text=text, structure_id=structure_id, locator_id=locator_id, note_id=note_id, **kwargs)

class AnnotationItem(BaseGraphItem):
    """Graph item representing an annotation."""
    def __init__(self, text, start, end, doc_id, structure_id, locator_id, note_id, same_as=None, type_=["Annotation"], **kwargs):
        super().__init__(type_=type_, text=text, structure_id=structure_id, locator_id=locator_id, note_id=note_id, same_as=same_as, **kwargs)
        if int(start)>-1:
            self.data.update({            
                "start": {"@value": int(start), "@type":"xsd:int"},
                "end": {"@value": int(end), "@type":"xsd:int"},
                "doc": {"@id": doc_id}
            })

class QuotationItem(BaseGraphItem):
    """Graph item representing a quotation block."""
    def __init__(self, text, structure_id, locator_id, note_id, type_="Quotation", **kwargs):
        super().__init__(type_=type_, text=text, structure_id=structure_id, locator_id=locator_id, note_id=note_id, **kwargs)
