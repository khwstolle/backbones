from importlib.metadata import version

project = "Backbones"
copyright = "2025, Kurt Stolle"
author = "Kurt Stolle"
release = version("backbones")
version = ".".join(release.split(".")[:2])
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.doctest',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.extlinks',
]
autodoc_typehints = "description"
numpydoc_show_class_members = False
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autoclass_content = "both"
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "alabaster"
html_static_path = ["_static"]
