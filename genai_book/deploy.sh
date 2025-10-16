#!/bin/bash
# Deploy Jupyter Book to GitHub Pages

set -e  # Exit on error

echo "Building Jupyter Book..."
jupyter-book build .

echo "Deploying to GitHub Pages..."
ghp-import -n -p -f _build/html

echo "✓ Deployment complete!"
echo "Your book will be available at: https://cjbarrie.github.io/GenAI_Soc/"
