from uuid import uuid4
from bs4 import BeautifulSoup, Tag
from lxml import etree
import re

def generate_uuid() -> str:
    """Generate a new UUID4 in URN format."""
    return f"urn:uuid:{uuid4()}"

def extract_text_lxml(html_snippet: str) -> str:
    """Extract plain text from an HTML snippet using lxml."""
    try:
        tree = etree.HTML(html_snippet)
        text = ''.join(tree.xpath('//text()')).strip()
        return re.sub(r'\s+', ' ', text)
    except Exception:
        return ""


def clean_html(html: str, mapping: dict, remove_empty_tags: bool = True) -> str:
    """Remove HTML elements mapped to 'IGNORE' and optionally remove empty tags."""
    soup = BeautifulSoup(html, "html.parser")

    ignore_tags = set()
    ignore_styles = set()
    ignore_patterns = mapping.get("IGNORE", {}).get("regex", [])
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    ignore_mapping = mapping.get("IGNORE", {})
    ignore_tags.update(ignore_mapping.get("tags", []))
    ignore_styles.update(ignore_mapping.get("styles", []))

    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue  # skip non-Tags (e.g., strings, comments)

        ignore = False

        if tag.name in ignore_tags:
            ignore = True

        if tag.attrs and tag.has_attr("style"):
            styles = {
                f"{k.strip()}:{v.strip()}"
                for k, v in (item.split(":") for item in tag["style"].split(";") if ":" in item)
            }
            if any(style in ignore_styles for style in styles):
                ignore = True

        if ignore:
            tag.decompose()

    for text_node in soup.find_all(string=True):
        parent = text_node.parent
        # nur sichtbarer Text, keine <script>, <style>, etc.
        if parent.name in {"script", "style", "head", "title", "meta"}:
            continue

        new_text = str(text_node)
        for pattern in ignore_patterns:
            new_text = re.sub(pattern, '', new_text)

        if new_text != text_node:
            text_node.replace_with(new_text)
    if remove_empty_tags:
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            text = (tag.get_text() or "").strip()
            if not text or text == '\xa0':
                tag.decompose()

    return str(soup)


def annotate_html_with_rdfa(html: str, mapping: dict) -> str:
    """
    Annotate an HTML string with RDFa 'typeof' attributes according to mapping.
    """
    from HTML_rdf_parser.parser import build_tag_style_lookup

    soup = BeautifulSoup(html, "html.parser")
    tag_lookup, style_lookup = build_tag_style_lookup(mapping)

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

        if match and not tag.has_attr("typeof"):
            typeof_value = match.get("types")
            if isinstance(typeof_value, list):
                typeof_value = " ".join(typeof_value)
            tag["typeof"] = typeof_value

    return str(soup)


def build_tag_style_lookup(mapping):
    """Build lookup tables for tags, styles, and optional classes, including regex-only definitions."""
    tag_lookup = {}
    style_lookup = {}

    for cls, config in mapping.items():
        if cls.startswith("@"):
            continue  # Skip @context, @type etc.

        if cls == "Annotation":
            for subtype, subconfig in config.items():
                tags = subconfig.get("tags", [])
                styles = subconfig.get("styles", [])
                regex = subconfig.get("regex")
                types = subconfig.get("types", subtype)
                class_name = subconfig.get("class")

                if isinstance(tags, str):
                    tags = [tags]
                if isinstance(styles, str):
                    styles = [styles]

                entry = {"class": "Annotation", "types": types, "class_name": class_name}

                for tag in tags:
                    tag_lookup.setdefault(tag, []).append(entry)
                for style in styles:
                    style_lookup.setdefault(style, []).append(entry)

                if regex:
                    tag_lookup.setdefault("span", []).append(entry)

        else:
            tags = config.get("tags", [])
            styles = config.get("styles", [])
            regex = config.get("regex")
            types = config.get("types", cls)
            class_name = config.get("class")

            if isinstance(tags, str):
                tags = [tags]
            if isinstance(styles, str):
                styles = [styles]

            entry = {"class": cls, "types": types, "class_name": class_name}

            for tag in tags:
                tag_lookup.setdefault(tag, []).append(entry)
            for style in styles:
                style_lookup.setdefault(style, []).append(entry)

            if regex:
                tag_lookup.setdefault("span", []).append(entry)

    return tag_lookup, style_lookup


