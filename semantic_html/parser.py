from semantic_html.models import *
from semantic_html.utils import *
from lxml import etree, html as lxml_html


def parse_note(html_input: str | etree._Element, mapping: dict, note_uri: str = None, metadata: dict = None, rdfa: bool = False, wadm: bool = False, tei: bool = False, conll: bool | dict = False, remove_empty_tags: bool = True) -> dict:
    """
    Parses a HTML note into a JSON-LD dictionary using lxml and xpath-based mapping.
    """
    # Validate html_input
    if isinstance(html_input, str):
        if not html_input.strip():
            raise ValueError("'html' is empty")
        tree = lxml_html.fromstring(html_input)
        # raw_html = html_input
    elif isinstance(html_input, etree._Element):
        tree = html_input
        # raw_html = etree.tostring(tree, encoding='unicode')
    else:
        raise TypeError(f"'html' must be a string or lxml Element, got {type(html_input).__name__}")

    # Validate mapping
    if not isinstance(mapping, dict):
        raise TypeError(f"'mapping' must be a dict, got {type(mapping).__name__}")
    if not mapping:
        raise ValueError("'mapping' is empty")

    # Extract special mapping entries
    metadata = metadata if metadata is not None else mapping.pop("metadata", None)
    wadm_meta = mapping.pop("wadm", None)
    note_uri = note_uri if note_uri else mapping.pop("@id", None)

    note_type = mapping.get('@type', 'Note')
    context = mapping.get('@context', DEFAULT_CONTEXT)
    same_as_xpath = mapping.pop('sameAs','self::a/@href | .//a/@href')
    tree = regex_wrap_tree(tree, mapping)

    # Clean the tree
    cleaned_tree = clean_html(tree, mapping, remove_empty_tags)
    cleaned_html = etree.tostring(cleaned_tree, encoding='unicode', method='html')
    # cleaned_html = etree.tostring(cleaned_tree, encoding='unicode', method='xml')

    note_text = extract_text_lxml(cleaned_tree)

    # Create root note item
    note_item = NoteItem(text=note_text, type_=note_type, html=cleaned_html,
                         metadata=metadata, wadm_meta=wadm_meta)
    if note_uri:
        note_item.data['@id'] = note_uri
    if conll: wadm = True
    items = [note_item.to_dict()]
    objects = [note_item]
    wadm_result = []
    

    # Track document and structure hierarchy
    current_structures = {}
    quotation_ids = {}
    doc_ids = {}
    doc_texts = {}
    current_structure = None
    current_locator = None
    base_id = note_item.data['@id']

    xpath_lookup = mapping_lookup(mapping)

    # Collect all node-entry matches
    matches = []  # list of (node, entries)
    for xp, entries in xpath_lookup.items():
        for node in cleaned_tree.xpath(xp):
            matches.append((node, entries, xp))
    # Determine document order for nodes
    order_map = {el: idx for idx, el in enumerate(cleaned_tree.iter())}
    order_index = 0
    # Sort matches by document order
    matches.sort(key=lambda pair: order_map.get(pair[0], float('inf')))
    # Iterate in document order

    # Iterate mapping entries with xpath
    for node, entries, xp in matches:
        same_as = get_same_as(node, same_as_xpath)
        node_tree_index = order_map.get(node, None)
        # Determine text
        if isinstance(node, etree._Element):
            text = extract_text_lxml(node)
            prefix, suffix = extract_context(node)                
            parent_iter = node.getparent()
            tag_name = node.tag
        else:
            # Handle attribute or text-node matches
            if hasattr(node, 'getparent'):
                # ElementUnicodeResult
                parent_iter = node.getparent()
                tag_name = parent_iter.tag if parent_iter is not None else None
                text = extract_text_lxml(node)
                prefix, suffix = extract_context(parent_iter)
            else:
                text = normalize_whitespace(str(node))
                prefix = ''
                suffix = ''
                parent_iter = None
                tag_name = None

        # Determine context offsets
        doc_id = base_id
        doc_text = note_text
        p = parent_iter
        parent_quotation_id = None

        while p is not None and (parent_quotation_id is None or doc_id == base_id):
            pid = id(p)
            if parent_quotation_id is None:
                parent_quotation_id = quotation_ids.get(pid)
            if doc_id == base_id:
                maybe_doc = doc_ids.get(pid)
                if maybe_doc is not None:
                    doc_id = maybe_doc
                    doc_text = doc_texts[pid]
            p = p.getparent()

        start, end = find_offset_with_context(text, prefix, suffix, doc_text)
        if start == -1 or end == -1:
            continue
        
        for m in entries:
            cls = m['class']
            types = m.get('types')

            split_pat = m.get('split')
            find_pat = m.get('find')
            regex = m.get('regex')

            re_matches = []

            if split_pat:
                split_parts = [p for p in re.split(split_pat, text) if p.strip()]
            else:
                split_parts = [text]

            search_pos = 0
            for part in split_parts:
                part_offset = text.find(part, search_pos)
                search_pos = part_offset + len(part)
                if find_pat:
                    for match in re.finditer(find_pat, part):
                        match_text = match.group()
                        local_start = part_offset + match.start()
                        re_matches.append((local_start, match_text))
                else:
                    re_matches.append((part_offset, part))

            if not re_matches:
                re_matches = [(0, text)]

            for local_start, match_text in re_matches:
                global_start = start + local_start
                global_end = global_start + len(match_text)
                selector = {
                    'start': global_start,
                    'end': global_end,
                    'prefix': doc_text[max(0, global_start - 30):global_start],
                    'suffix': doc_text[global_end:global_end + 30],
                    'xpath': xp
                }
                if regex:
                    selector['regex'] = regex

                # Build items by class
                if cls == 'Document':
                    order_index += 1                    
                    item = DocItem(text=match_text, structure_id=current_structure,
                                    locator_id=current_locator, note_id=base_id,
                                    doc_id=doc_id, type_=types, selector=selector,
                                    quotation_id=parent_quotation_id,
                                    metadata=metadata, wadm_meta=wadm_meta, tree_index = node_tree_index, order_index = order_index)
                    items.append(item.to_dict())
                    if tei: objects.append(item)
                    if wadm: wadm_result.append(item.to_wadm())
                    doc_ids[id(node)] = item.data['@id']
                    doc_texts[id(node)] = item.data['text']

                elif cls == 'Locator':
                    order_index += 1
                    item = LocatorItem(text=match_text, structure_id=current_structure,
                                        note_id=base_id, doc_id=doc_id, same_as=same_as,
                                        type_=types, selector=selector,
                                        quotation_id=parent_quotation_id,
                                        metadata=metadata, wadm_meta=wadm_meta, tree_index = node_tree_index, order_index = order_index)
                    items.append(item.to_dict())
                    if tei: objects.append(item)
                    if wadm: wadm_result.append(item.to_wadm())
                    current_locator = item.data['@id']

                elif cls == 'Structure':
                    order_index += 1                    
                    level = int(tag_name[1]) if tag_name and tag_name.startswith('h') and tag_name[1].isdigit() else 1
                    parent_id = None
                    for lvl in range(level-1, 0, -1):
                        if lvl in current_structures:
                            parent_id = current_structures[lvl]
                            break
                    item = StructureItem(text=match_text, level=level, note_id=base_id,
                                            doc_id=doc_id, type_=types,
                                            structure_id=parent_id, same_as=same_as,
                                            locator_id=current_locator,
                                            quotation_id=parent_quotation_id,
                                            selector=selector,
                                            metadata=metadata, wadm_meta=wadm_meta, tree_index = node_tree_index, order_index = order_index)
                    items.append(item.to_dict())
                    if tei: objects.append(item)
                    if wadm: wadm_result.append(item.to_wadm())
                    current_structures[level] = item.data['@id']
                    current_structure = item.data['@id']

                elif cls == 'Quotation':
                    order_index += 1
                    item = QuotationItem(text=match_text, structure_id=current_structure,
                                            doc_id=doc_id, locator_id=current_locator, quotation_id=parent_quotation_id,
                                            note_id=base_id, type_=types,
                                            selector=selector, metadata=metadata,
                                            wadm_meta=wadm_meta, tree_index = node_tree_index, order_index = order_index)
                    items.append(item.to_dict())
                    if tei: objects.append(item)
                    quotation_ids[id(node)] = item.data['@id']
                    if wadm: wadm_result.append(item.to_wadm())

                elif cls == 'Annotation':
                    order_index += 1
                    item = AnnotationItem(text=match_text, doc_id=doc_id,
                                            structure_id=current_structure,
                                            locator_id=current_locator,
                                            quotation_id=parent_quotation_id,
                                            note_id=base_id, same_as=same_as,
                                            type_=types, selector=selector,
                                            metadata=metadata, wadm_meta=wadm_meta, tree_index = node_tree_index, order_index = order_index)
                    items.append(item.to_dict())
                    if tei: objects.append(item)
                    if wadm: wadm_result.append(item.to_wadm())

    # Build results
    jsonld = {'@context': context, '@graph': items}
    result = {'MAP': mapping, 'JSON-LD': jsonld}
    
    if wadm:
        result['WADM'] = {'@context': wadm_meta.get('@context', WADM_CONTEXT) if wadm_meta else WADM_CONTEXT,
                          '@graph': wadm_result}
        
    if rdfa:
        rdfa_tree = annotate_tree_with_rdfa(cleaned_tree, mapping, context)
        result['RDFa'] = etree.tostring(rdfa_tree, encoding='unicode', method='html')

    if tei:
        tei_tree = build_tei_from_items(objects)
        result['TEI'] = etree.tostring(
            tei_tree, encoding='utf-8',
            xml_declaration=True, pretty_print=True
        ).decode('utf-8')

    if conll:
        config = conll if isinstance(conll, dict) else None
        result['CoNLL'] = wadm_to_conll(wadm_result, config, jsonld=jsonld)

    return result