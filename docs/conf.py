import os
import sys
sys.path.insert(0, os.path.abspath('../semantic_html'))

extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.napoleon'
]

autoapi_type = 'python'
autoapi_dirs = ['../semantic_html']

project = 'Semantic HTML'
html_title = 'Semantic HTML Utility Documentation'

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'

source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}