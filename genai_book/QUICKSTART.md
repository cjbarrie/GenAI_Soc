# Quick Start Guide

Get your Jupyter Book online in 3 steps!

## Step 1: Install Jupyter Book

```bash
cd genai_book
pip install -r requirements.txt
```

## Step 2: Build Locally (Optional)

Preview your book before publishing:

```bash
jupyter-book build .
```

Open `_build/html/index.html` in your browser.

## Step 3: Deploy to GitHub Pages

### Automatic Deployment (Recommended)

The repository includes a GitHub Actions workflow that automatically builds and deploys your book when you push to the `main` branch.

Just commit and push:

```bash
git add .
git commit -m "Add Jupyter Book"
git push origin main
```

Your book will be live at: `https://cjbarrie.github.io/GenAI_Soc/`

### Manual Deployment

If you prefer manual control:

```bash
cd genai_book
./deploy.sh
```

Or step-by-step:

```bash
jupyter-book build .
ghp-import -n -p -f _build/html
```

## Troubleshooting

### "ghp-import: command not found"

Install it:
```bash
pip install ghp-import
```

### "GitHub Pages not enabled"

1. Go to your repo settings on GitHub
2. Navigate to "Pages"
3. Set source to "gh-pages" branch
4. Save

### Build errors

Clean and rebuild:
```bash
jupyter-book clean .
jupyter-book build .
```

## Customization

### Change Book Title

Edit `_config.yml`:
```yaml
title: Your Book Title
author: Your Name
```

### Add Chapters

1. Add `.ipynb` or `.md` file to `genai_book/`
2. Update `_toc.yml`:
```yaml
chapters:
  - file: your_new_chapter
    title: "Your Chapter Title"
```

### Change Theme

Edit `_config.yml`:
```yaml
format:
  html:
    theme: sphinx_book_theme  # or cosmo, darkly, etc.
```

## Alternative Hosting

### Netlify

1. Build: `jupyter-book build .`
2. Drag `_build/html` to [Netlify](https://app.netlify.com/drop)
3. Get instant URL

### Read the Docs

1. Connect repo to [Read the Docs](https://readthedocs.org/)
2. Configure build directory as `genai_book`
3. Automatic builds on push

## Next Steps

- Add more notebooks
- Customize styling
- Add interactive widgets
- Enable comments (hypothesis, utterances)
- Add Google Analytics

See [Jupyter Book documentation](https://jupyterbook.org/) for more.
