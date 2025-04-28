from HTML_rdf_parser.models import *
from HTML_rdf_parser.utils import *
from bs4 import BeautifulSoup

def parse_note(html: str, mapping: dict, note_uri: str = None, return_annotated_html: bool = False) -> dict:
    """
    Parses a HTML note HTML string into a JSON-LD dictionary (optionally also annotated HTML).

    Args:
        html (str): The HTML content of the HTML note.
        mapping (dict): A dictionary mapping classes, tags, styles, and types.
        note_uri (str, optional): If provided, used as the Note's @id.
        return_annotated_html (bool, optional): If True, also return RDFa-annotated HTML.

    Returns:
        dict: JSON-LD structured data or dict with jsonld and annotated_html.
    """
    
    tag_lookup, style_lookup = build_tag_style_lookup(mapping)
    items = []

    html_cleaned = clean_html(html, mapping)
    soup = BeautifulSoup(html_cleaned, "html.parser")
    note_text = extract_text_lxml(html_cleaned)
    note_type = mapping.get('@type', 'Note')

    note_item = NoteItem(note_text, type_=note_type)
    if note_uri:
        note_item.data["@id"] = note_uri
    items.append(note_item.to_dict())

    current_structures_by_level = {}
    current_structure = None
    current_locator = None
    current_doc_id = None
    current_doc_text = None

    for tag in soup.find_all(True):
        tag_name = tag.name
        match = tag_lookup.get(tag_name)

        if not match and tag.has_attr("style"):
            styles = [
                f"{k.strip()}:{v.strip()}"
                for k, v in (item.split(":") for item in tag["style"].split(";") if ":" in item)
            ]
            for style in styles:
                match = style_lookup.get(style)
                if match:
                    break

        if not match:
            continue

        cls = match["class"]
        types = match.get("types")
        note_id = note_item.data["@id"]
        text = extract_text_lxml(str(tag))        

        if cls == "Document":
            doc_item = DocItem(text, structure_id=current_structure, locator_id=current_locator, note_id=note_id, type_=types)            
            items.append(doc_item.to_dict())
            current_doc_id  = doc_item.data["@id"]
            current_doc_text  = doc_item.data["text"]["@value"]
        elif cls == "Locator":
            locator_item = LocatorItem(text, structure_id=current_structure, note_id=note_id, type_=types)
            items.append(locator_item.to_dict())
            current_locator = locator_item.data["@id"]
        elif cls == "Structure":
            level = int(tag.name[1]) if tag.name[1].isdigit() else 1
            parent_structure_id = None

            for parent_level in reversed(range(1, level)):
                if parent_level in current_structures_by_level:
                    parent_structure_id = current_structures_by_level[parent_level]
                    break

            structure_item = StructureItem(
                text=text,
                level=level,
                note_id=note_id,
                type_=types,
                structure_id=parent_structure_id,
                locator_id=current_locator
            )
            items.append(structure_item.to_dict())

            current_structures_by_level[level] = structure_item.data["@id"]
            current_structure = structure_item.data["@id"]
        elif cls == "Quotation":
            quotation_item = QuotationItem(text, structure_id=current_structure, locator_id=current_locator, note_id=note_id, type_=types)
            items.append(quotation_item.to_dict())
        elif cls == "Annotation":
            if current_doc_text and text:
                start = current_doc_text.find(text)
            else:
                start = -1
            end = start + len(text)
            same_as = None
            link_tag = tag.find("a")
            if link_tag and link_tag.has_attr("href"):
                same_as = link_tag["href"]

            annotation_item = AnnotationItem(
                text=text,
                start=start,
                end=end,
                doc_id = current_doc_id, # if current_doc_id else note_id,
                structure_id=current_structure,
                locator_id=current_locator,
                note_id=note_id,
                same_as=same_as, 
                type_=types
            )
            items.append(annotation_item.to_dict())

    # for idx, (tag, structure_id) in enumerate(structure_lookup):
    #     if idx > 0:
    #         items[idx+1]["hasParentStructure"] = {"@id": structure_lookup[idx-1][1]}

    context = mapping.get('@context', None)

    jsonld_result = {
        "@graph": items
    }
    if context:
        jsonld_result["@context"] = context

    if return_annotated_html:
        annotated_html = annotate_html_with_rdfa(html_cleaned, mapping)
        return {
            "jsonld": jsonld_result,
            "RDFa": annotated_html
        }
    else:
        return jsonld_result