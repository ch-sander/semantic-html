from uuid import uuid4
from lxml import etree, html as lxml_html
import re

def generate_uuid() -> str:
    """Generate a new UUID4 in URN format."""
    return f"urn:uuid:{uuid4()}"

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_text_lxml(node) -> str:
    """Extract plain text using lxml."""
    try:
        if isinstance(node, str):
            tree = lxml_html.fromstring(node)
            texts = tree.xpath('//text()')
        else:
            # lxml element
            texts = node.xpath('.//text()')
        return normalize_whitespace(''.join(texts))
    except Exception:
        return ""

def find_offset_with_context(text, prefix, suffix, doc_text, max_chars=30): 
    pattern = re.compile(re.escape(text))
    for match in pattern.finditer(doc_text):
        start, end = match.start(), match.end()
        doc_prefix = doc_text[max(0, start - max_chars):start].strip()
        doc_suffix = doc_text[end:end + max_chars].strip()
        if doc_prefix.endswith(prefix.strip()) and doc_suffix.startswith(suffix.strip()):
            return start, end
    print(f"WARNING: Could not match string {text} in {doc_text}")
    return -1, -1

def extract_context(node: etree._Element, max_chars: int = 30) -> tuple[str, str]:
    """
    Extract prefix and suffix context around the node's text within its parent text.
    """
    # Get full normalized text of the node and its parent
    text = normalize_whitespace(''.join(node.itertext()))
    parent = node.getparent()
    if parent is None:
        return "", ""
    parent_text = normalize_whitespace(''.join(parent.itertext()))
    idx = parent_text.find(text)

    if idx == -1:
        return "", ""

    prefix = parent_text[max(0, idx - max_chars):idx]
    suffix = parent_text[idx + len(text):idx + len(text) + max_chars]
    return prefix, suffix

def clean_html(tree: etree._Element, mapping: dict, remove_empty_tags: bool = True) -> etree._Element:
    """
    Remove HTML elements mapped to 'IGNORE' via xpath and optionally remove empty tags using lxml.
    """
    ignore_cfg = mapping.get("IGNORE", {})
    xpaths = ignore_cfg.get("xpath") or []
    if isinstance(xpaths, str):
        xpaths = [xpaths]

    # Remove nodes matching ignore xpaths
    for xp in xpaths:
        for node in tree.xpath(xp):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    # Clean text and tail via regex in ignore regex
    import re
    patterns = ignore_cfg.get("regex", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    for node in tree.iter():
        if node.text:
            txt = node.text
            for pat in patterns:
                txt = re.sub(pat, '', txt)
            node.text = txt
        if node.tail:
            tail = node.tail
            for pat in patterns:
                tail = re.sub(pat, '', tail)
            node.tail = tail

    # Optionally remove empty tags
    if remove_empty_tags:
        for elem in list(tree.iter()):
            if not isinstance(elem.tag, str):
                continue
            if not ''.join(elem.itertext()).strip() and len(elem) == 0:
                parent = elem.getparent()
                if parent is not None:
                    parent.remove(elem)
    return tree

def annotate_tree_with_rdfa(tree: etree._Element, mapping: dict, context: dict = None) -> etree._Element:
    """
    Annotate an lxml tree with RDFa 'typeof' attributes and namespace declarations based on mapping @context.
    """
    # Read RDFa context for namespace declarations
    context_map = context or mapping.get('@context', {}) or {}
    # Determine root element to attach namespace declarations
    root = tree.getroottree().getroot() if hasattr(tree, 'getroottree') else tree
    if isinstance(context_map, dict):
        for prefix, uri in context_map.items():
            # Resolve JSON-LD context entries to URI strings
            if isinstance(uri, dict):
                uri = uri.get('@id') or uri.get('id')
            if not isinstance(uri, str):
                continue
            # Only declare actual namespace URIs (skip CURIE references)
            if not uri.startswith('http'):
                continue
            # Handle vocab and base
            if prefix == '@vocab':
                root.set('vocab', uri)
                continue
            if prefix == '@base':
                root.set('xml:base', uri)
                continue
            # Skip other JSON-LD keywords
            if prefix.startswith('@'):
                continue
            # Set namespace declaration
            root.set(f'xmlns:{prefix}', uri)
    # Build lookup of xpaths to semantic types
    xpath_lookup = mapping_lookup(mapping)
    # For each xpath and entries, set typeof on matching elements
    for xp, entries in xpath_lookup.items():
        for node in tree.xpath(xp):
            if not isinstance(node, etree._Element):
                continue
            # If already has typeof, skip
            if node.get('typeof'):
                continue
            # Compute typeof CURIE or list
            types = entries[0].get('types')
            if isinstance(types, list):
                typeof_val = ' '.join(types)
            else:
                typeof_val = types
            node.set('typeof', typeof_val)
    return tree


def mapping_lookup(mapping):
    """
    Build lookup table for xpath entries in mapping.
    Handles string or list of xpaths, and special Annotation subtypes.
    Returns xpath_lookup dict.
    """
    xpath_lookup = {}
    for cls, config in mapping.items():
        if cls.startswith("@"):  # skip metadata entries
            continue
        if cls == "Annotation" and isinstance(config, dict):
            for subtype, subconfig in config.items():
                pats = subconfig.get('xpath')
                if not pats:
                    continue
                xp_list = pats if isinstance(pats, list) else [pats]
                entry = {
                    'class': 'Annotation',
                    'types': subconfig.get('types', subtype),
                    'regex': subconfig.get('regex')
                }
                for xp in xp_list:
                    xpath_lookup.setdefault(xp, []).append(entry)
        elif isinstance(config, dict):
            pats = config.get('xpath')
            if not pats:
                continue
            xp_list = pats if isinstance(pats, list) else [pats]
            for xp in xp_list:
                entry = {
                    'class': cls,
                    'types': config.get('types', cls),
                    'regex': config.get('regex')
                }
                xpath_lookup.setdefault(xp, []).append(entry)
    return xpath_lookup

def regex_wrap_tree(tree: etree._Element, mapping: dict) -> etree._Element:
    """
    Wrap regex matches in spans based on mapping regex entries by serializing and reparsing.
    """
    import re
    # Collect patterns
    patterns = []  # list of (pattern, class)
    for cls, cfg in mapping.items():
        if cls.startswith("@") or cls == "IGNORE":
            continue
        if cls == "Annotation":
            for subtype, subcfg in cfg.items():
                pats = subcfg.get("regex")
                if not pats:
                    continue
                if isinstance(pats, str):
                    pats = [pats]
                for pat in pats:
                    patterns.append((pat, cls))
        else:
            pats = cfg.get("regex")
            if not pats:
                continue
            if isinstance(pats, str):
                pats = [pats]
            for pat in pats:
                patterns.append((pat, cls))
    if not patterns:
        return tree
    # Serialize to HTML
    raw_html = etree.tostring(tree, encoding='unicode', method='html')
    # Apply regex wraps on entire HTML
    for pat, cls in patterns:
        repl = lambda m, cls=cls: f'<span class="{cls}">{m.group(0)}</span>'
        raw_html = re.sub(pat, repl, raw_html)
    # Reparse into tree
    return lxml_html.fromstring(raw_html)