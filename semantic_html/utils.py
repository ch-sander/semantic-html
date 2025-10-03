from uuid import uuid4
from lxml import etree, html as lxml_html
import re

def generate_uuid() -> str:
    """Generate a new UUID4 in URN format."""
    return f"urn:uuid:{uuid4()}"

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def get_same_as(node: etree._Element) -> str:
    same_as = None
    if isinstance(node, etree._Element):
        if node.tag == 'a' and node.get('href'):
            same_as = node.get('href')
        else:
            link = node.find('.//a')
            if link is not None and link.get('href'):
                same_as = link.get('href')
    return same_as

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
                    'regex': subconfig.get('regex'),
                    'split': subconfig.get('split'),
                    'find': subconfig.get('find')
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
                    'regex': config.get('regex'),
                    'split': config.get('split'),
                    'find': config.get('find')
                }
                xpath_lookup.setdefault(xp, []).append(entry)
    return xpath_lookup

def regex_wrap_tree(tree: etree._Element, mapping: dict) -> etree._Element:
    """
    Traverses the lxml tree and wraps regex matches in <span> tags,
    mutates mapping.
    """
    entries = []

    def _collect(subm: dict):
        for key, v in subm.items():
            if key == "IGNORE" or key.startswith("@"):
                continue
            if isinstance(v, dict):
                pats = v.get("regex")
                if pats:
                    # normalize patterns to list
                    if isinstance(pats, str):
                        pats = [pats]

                    # class-name and uuid4
                    cls = v.get("class", key)
                    v["class"] = cls
                    rid = uuid4().hex
                    v["rw_uuid"] = rid

                    # XPath data-rw-uuid
                    v["xpath"] = f".//span[@data-rw-uuid='{rid}']"

                    for pat in pats:
                        entries.append((re.compile(pat), cls, rid))

                _collect(v)

    _collect(mapping)
    if not entries:
        return tree

    def _split_and_wrap(text: str, pattern: re.Pattern, cls: str, rid: str):
        parts, last = [], 0
        for m in pattern.finditer(text):
            if m.start() > last:
                parts.append(text[last:m.start()])
            span = etree.Element("span", {
                "class": cls,
                "data-rw-uuid": rid
            })
            span.text = m.group(0)
            parts.append(span)
            last = m.end()
        if last < len(text):
            parts.append(text[last:])
        return parts if len(parts) > 1 else None

    def _apply(node: etree._Element, pattern: re.Pattern, cls: str, rid: str):
        # node.text
        if node.text:
            wrapped = _split_and_wrap(node.text, pattern, cls, rid)
            if wrapped:
                node.text = ""
                for part in wrapped:
                    if isinstance(part, str):
                        if len(node):
                            node[-1].tail = (node[-1].tail or "") + part
                        else:
                            node.text += part
                    else:
                        node.append(part)

        # Children and their tail
        for child in list(node):
            if child.tail:
                wrapped = _split_and_wrap(child.tail, pattern, cls, rid)
                if wrapped:
                    child.tail = ""
                    idx = list(node).index(child)
                    for part in wrapped:
                        if isinstance(part, str):
                            if idx >= 0:
                                node[idx].tail = (node[idx].tail or "") + part
                            else:
                                node.text = (node.text or "") + part
                        else:
                            idx += 1
                            node.insert(idx, part)
            _apply(child, pattern, cls, rid)

    # 4 Apply to all patterns
    for pattern, cls, rid in entries:
        _apply(tree, pattern, cls, rid)

    return tree

def tokenize(text):
    """Simple and robust tokenization."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def build_token_spans(text, tokens):
    """Compute start and end character offsets for each token."""
    spans = []
    offset = 0
    for tok in tokens:
        start = text.find(tok, offset)
        if start == -1:
            raise ValueError(f"Token '{tok}' not found in text after {offset}")
        end = start + len(tok)
        spans.append((start, end))
        offset = end
    return spans


# ---------------- Selector Normalization ----------------

def _flatten_selector(sel):
    """
    Handle nested selector structures like refinedBy, Choice, List.
    Always return a list of flat selectors.
    """
    out = []
    if not isinstance(sel, dict):
        return out

    stype = sel.get("type")

    if stype in ("TextPositionSelector", "TextQuoteSelector"):
        out.append(sel)

    elif "refinedBy" in sel:
        out.append(sel)
        out.extend(_flatten_selector(sel["refinedBy"]))

    elif stype == "Choice":
        for option in sel.get("items", []):
            out.extend(_flatten_selector(option))

    elif stype == "List":
        for option in sel.get("members", []):
            out.extend(_flatten_selector(option))

    return out


def normalize_wadm(wadm, whitelist=None, blacklist=None):
    """
    Normalize WADM annotations into a simpler structure:
    - entity_type: string (from body purpose: tagging)
    - selectors: list of selector dicts
    """
    normalized = []

    annotations = wadm.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = [annotations]

    for ann in annotations:
        # --- Extract entity type ---
        entity_type = None
        bodies = ann.get("body", [])
        if not isinstance(bodies, list):
            bodies = [bodies]

        for b in bodies:
            if isinstance(b, dict) and b.get("purpose") == "tagging":
                entity_type = b.get("value")
                break
            if isinstance(b, str):
                entity_type = b.split("/")[-1]

        if not entity_type:
            continue

        # Apply whitelist/blacklist
        if whitelist is not None and entity_type not in whitelist:
            continue
        if blacklist is not None and entity_type in blacklist:
            continue

        # --- Extract selectors ---
        selectors = []
        targets = ann.get("target", [])
        if not isinstance(targets, list):
            targets = [targets]

        for t in targets:
            if isinstance(t, dict):
                if "selector" in t:
                    sels = t["selector"]
                    if not isinstance(sels, list):
                        sels = [sels]
                    for s in sels:
                        selectors.extend(_flatten_selector(s))

        if selectors:
            normalized.append({"entity_type": entity_type, "selectors": selectors})

    return normalized


# ---------------- JSON-LD Text Resolver ----------------

def resolve_texts(jsonld):
    """Build a lookup from @id → text for all resources in a JSON-LD @graph."""
    lookup = {}
    for res in jsonld.get("@graph", []):
        if "@id" in res and "text" in res:
            lookup[res["@id"]] = res["text"]
    return lookup


# ---------------- CoNLL Conversion ----------------

def _conll_from_annotations(text, annotations, max_span_tokens=None):
    """Helper: map annotations to BIO labels for a given text."""
    tokens = tokenize(text)
    spans = build_token_spans(text, tokens)
    labels = ["O"] * len(tokens)

    for ann in annotations:
        entity_type = ann["entity_type"]
        for sel in ann["selectors"]:
            start, end = None, None
            if sel.get("type") == "TextPositionSelector":
                start, end = sel.get("start"), sel.get("end")
            elif sel.get("type") == "TextQuoteSelector":
                exact = sel.get("exact")
                if exact:
                    start = text.find(exact)
                    if start != -1:
                        end = start + len(exact)

            if start is None or end is None:
                continue

            token_indices = [i for i, (s, e) in enumerate(spans)
                             if not (e <= start or s >= end)]
            if not token_indices:
                continue

            # wenn max_span_tokens gesetzt ist → ganze Entity verwerfen, falls zu lang
            if max_span_tokens is not None and len(token_indices) > max_span_tokens:
                continue

            inside = False
            for i in token_indices:
                tag = "B-" + entity_type if not inside else "I-" + entity_type
                labels[i] = tag
                inside = True

    return [[(tok, lab) for tok, lab in zip(tokens, labels)]]



def conll_to_string(sentences):
    """Return sentences in CoNLL format as a string."""
    lines = []
    for sent in sentences:
        for tok, lab in sent:
            lines.append(f"{tok}\t{lab}")
        lines.append("")  # sentence separator
    return "\n".join(lines)