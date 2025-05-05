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
    regex_wrapper = RegexWrapper(mapping)
    html = regex_wrapper.wrap(html)
    tag_lookup, style_lookup = build_tag_style_lookup(mapping)
    items = []
    html_cleaned = clean_html(html, mapping)
    soup = BeautifulSoup(html_cleaned, "html.parser")
    note_text = extract_text_lxml(html_cleaned)
    note_type = mapping.get('@type', 'Note')
    note_item = NoteItem(note_text, type_=note_type, html = html_cleaned)
    if note_uri:
        note_item.data["@id"] = note_uri
    items.append(note_item.to_dict())

    current_structures_by_level = {}
    doc_ids_by_tag = {}
    doc_texts_by_tag = {}
    current_structure = None
    current_locator = None


    for tag in soup.find_all(True):
        tag_name = tag.name
        matches = tag_lookup.get(tag_name, [])
        if not isinstance(matches, list):
            matches = [matches]

        match = None
        for m in matches:
            expected_class = m.get("class_name")
            if expected_class:
                if tag.has_attr("class") and expected_class in tag.get("class", []):
                    match = m
                    break
            else:
                match = m
                break

        if not match and tag.has_attr("style"):
            styles = [
                f"{k.strip()}:{v.strip()}"
                for k, v in (item.split(":") for item in tag["style"].split(";") if ":" in item)
            ]
            for style in styles:
                style_matches = style_lookup.get(style, [])
                if not isinstance(style_matches, list):
                    style_matches = [style_matches]

                for m in style_matches:
                    expected_class = m.get("class_name")
                    if expected_class:
                        if tag.has_attr("class") and expected_class in tag.get("class", []):
                            match = m
                            break
                    else:
                        match = m
                        break
                if match:
                    break

        if not match:
            continue

        expected_class = match.get("class_name")
        if expected_class:
            if not (tag.has_attr("class") and expected_class in tag.get("class", [])):
                continue

        cls = match["class"]
        types = match.get("types")
        note_id = note_item.data["@id"]
        text = extract_text_lxml(str(tag))        

        if cls == "Document":
            doc_item = DocItem(text, structure_id=current_structure, locator_id=current_locator, note_id=note_id, type_=types)            
            items.append(doc_item.to_dict())
            doc_ids_by_tag[id(tag)] = doc_item.data["@id"]
            doc_texts_by_tag[id(tag)] = doc_item.data["text"]
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
            # Determine if this annotation is within a Document tag
            doc_id = None
            start = -1
            end = -1
            # Search ancestors for a Document tag
            for parent in tag.parents:
                if id(parent) in doc_ids_by_tag:
                    doc_id = doc_ids_by_tag[id(parent)]
                    doc_text = doc_texts_by_tag[id(parent)]
                    # Compute offsets only if text exists in this document
                    if text and doc_text:
                        start = doc_text.find(text)
                        if start >= 0:
                            end = start + len(text)
                    break

            same_as = None
            if tag.name == "a" and tag.has_attr("href"):
                same_as = tag["href"]
            else:
                link_tag = tag.find("a")
                if link_tag and link_tag.has_attr("href"):
                    same_as = link_tag["href"]

            # Build annotation, include doc_id only if found
            annotation_item = AnnotationItem(
                text=text,
                start=start,
                end=end,
                doc_id=doc_id,
                structure_id=current_structure,
                locator_id=current_locator,
                note_id=note_id,
                same_as=same_as,
                type_=types
            )
            items.append(annotation_item.to_dict())

    context = mapping.get('@context', DEFAULT_CONTEXT)

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