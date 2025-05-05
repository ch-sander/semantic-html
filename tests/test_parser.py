from semantic_html.parser import parse_note

def test_parse_note_minimal():
    html = """
    <!DOCTYPE html><html><head><meta charset="utf-8"></head><body><div class="zotero-notes"><div class="zotero-note"><h1>Header1</h1>
    <h2>header2</h2>
    <h3>header3</h3>
    <p><code>p.44</code></p>
    <p>comment related to p. 44 and header 3</p>
    <blockquote>
    <p style="margin-left: 30px;">“<em>Man</em> is a concept. <strong><a href="wd:Paris" rel="noopener noreferrer nofollow">Paris</a></strong> is a <em>city</em>. that is testified by <u>wikipedia</u>.”</p>
    </blockquote>
    <p><span class="citation">(<a href="zotero://select/library/items/KUHYUVEH">“test”</a>)</span></p>
    <h3>header4</h3>
    <p><code>p.54</code></p>
    <p>comment related to p. 54 and header 4</p>
    <blockquote>
    <p style="margin-left: 30px;">“<em>Woman </em>is a concept. <strong><a href="wd:Rome" rel="noopener noreferrer nofollow">Rome</a> </strong>is a <em>city</em>. that is testified by <u>wikipedia</u>.”</p>
    </blockquote>
    <pre>This will be ignored</pre>
    </div></div></body></html>
    """

    mapping = {
        "Note": {
            "types": ["Note"]
        },
        "Doc": {
            "tags": ["p"],
            "types": ["Doc"]
        },
        "Annotation": {
            "Concept": {
                "tags": ["em"],
                "styles": ["font-style:italic"],
                "types": ["Annotation","Concept"]
            },
            "Entity": {
                "tags": ["strong"],
                "styles": ["font-weight:bold"],
                "types": ["Entity"]
            },
            "Reference": {
                "tags": ["u"],
                "styles": ["text-decoration:underline"],
                "types": ["Reference"]
            }
        },
        "Locator": {
            "tags": ["code"],
            "types": ["Page-Locator"]
        },
        "Structure": {
            "tags": ["h1", "h2", "h3"],
            "types": ["Structure"]
        },
        "Quotation": {
            "tags": ["blockquote"],
            "types": ["Quotation"]
        },
        "IGNORE": {
            "tags": ["pre"]
        },
        "@context": {
            "@vocab": "https://semantic-zotero.org/vocab#",
            "same:as": {"@id": "owl:sameAs", "@type": "@id"},
            "locator": {"@id": "in_locator", "@type": "@id"},
            "structure": {"@id": "in_structure", "@type": "@id"},
            "note": {"@id": "in_note", "@type": "@id"}
            },
        "@type": ["ResearchNote"]
    }
    result = parse_note(html, mapping)
    assert "@context" in result
    assert "@graph" in result
    assert isinstance(result["@graph"], list)
