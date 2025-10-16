# GenAI for Social Science - Jupyter Book

This directory contains a Jupyter Book that compiles all the computational notebooks for the Generative AI and Society course.

## Building the Book

### Install Jupyter Book

```bash
pip install -r requirements.txt
```

### Build HTML

```bash
jupyter-book build .
```

The HTML will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build

```bash
jupyter-book clean .
```

## Publishing Online

### Option 1: GitHub Pages (Recommended)

1. Build the book:
```bash
jupyter-book build .
```

2. Push to GitHub Pages:
```bash
ghp-import -n -p -f _build/html
```

This will publish to `https://cjbarrie.github.io/GenAI_Soc/`

### Option 2: Netlify

1. Build the book
2. Drag the `_build/html` folder to Netlify
3. Get a custom URL

### Option 3: Read the Docs

1. Connect your GitHub repo to Read the Docs
2. Configure to build from the `genai_book` directory
3. Automatic builds on each commit

## Structure

```
genai_book/
├── _config.yml          # Book configuration
├── _toc.yml             # Table of contents
├── intro.md             # Landing page
├── requirements.txt     # Python dependencies
├── references.bib       # Bibliography
├── week2_embeddings.ipynb
├── week3_transformers.ipynb
├── week4_llms.ipynb
├── week5_qualitative.ipynb
└── week6_annotation.ipynb
```

## Adding New Content

1. Add notebook to `genai_book/`
2. Update `_toc.yml` to include it
3. Rebuild: `jupyter-book build .`

## Customization

Edit `_config.yml` to customize:
- Book title and author
- Repository links
- Launch buttons (Colab, Binder)
- Theme and styling
