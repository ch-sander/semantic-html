from uuid import uuid4
from bs4 import BeautifulSoup, Tag
from lxml import etree

def generate_uuid() -> str:
    """Generate a new UUID in URN format."""
    return f"urn:uuid:{uuid4()}"

def extract_text_lxml(html_snippet: str) -> str:
    """Extract plain text from an HTML snippet using lxml."""
    try:
        tree = etree.HTML(html_snippet)
        return ''.join(tree.xpath('//text()')).strip()
    except Exception:
        return ""


def clean_html(html: str, mapping: dict, remove_empty_tags: bool = True) -> str:
    """Remove HTML elements mapped to 'IGNORE' and optionally remove empty tags."""
    soup = BeautifulSoup(html, "html.parser")

    ignore_tags = set()
    ignore_styles = set()
    if "IGNORE" in mapping:
        ignore_mapping = mapping["IGNORE"]
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

        if match and not tag.has_attr("typeof"):
            typeof_value = match.get("types")
            if isinstance(typeof_value, list):
                typeof_value = " ".join(typeof_value)
            tag["typeof"] = typeof_value

    return str(soup)

def build_tag_style_lookup(mapping):
    """Build lookup tables for tags and styles."""
    tag_lookup = {}
    style_lookup = {}
    for cls, config in mapping.items():
        if cls.startswith("@"):
            continue  # Skip @context, @type etc.
        if cls == "Annotation":
            for subtype, subconfig in config.items():
                tags = subconfig.get("tags", [])
                styles = subconfig.get("styles", [])
                types = subconfig.get("types", subtype)
                if isinstance(tags, str):
                    tags = [tags]
                if isinstance(styles, str):
                    styles = [styles]
                for tag in tags:
                    tag_lookup[tag] = {"class": "Annotation", "types": types}
                for style in styles:
                    style_lookup[style] = {"class": "Annotation", "types": types}
        else:
            tags = config.get("tags", [])
            styles = config.get("styles", [])
            types = config.get("types", cls)
            if isinstance(tags, str):
                tags = [tags]
            if isinstance(styles, str):
                styles = [styles]
            for tag in tags:
                tag_lookup[tag] = {"class": cls, "types": types}
            for style in styles:
                style_lookup[style] = {"class": cls, "types": types}
    return tag_lookup, style_lookup
