# Documentation Source Files

This directory contains the source files for the Finetune CLI documentation.

## Structure

```
docs/
├── index.md              # Homepage
├── installation.md       # Installation guide
├── usage.md             # Usage guide
├── configuration.md     # Configuration reference
├── examples.md          # Practical examples
├── api.md              # API reference
├── troubleshooting.md  # Troubleshooting guide
├── stylesheets/
│   └── extra.css       # Custom CSS styles
└── README.md           # This file
```

## Building Documentation Locally

### Prerequisites

Install MkDocs and dependencies:

```bash
pip install mkdocs-material mkdocs-minify-plugin pymdown-extensions
```

### Local Development Server

Run a local server to preview changes:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

The server will automatically reload when you save changes to the documentation files.

### Building Static Site

Generate static HTML files:

```bash
mkdocs build
```

Output will be in the `site/` directory.

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

The deployment workflow is defined in `.github/workflows/deploy_docs.yml`.

### Manual Deployment

To manually deploy to GitHub Pages:

```bash
mkdocs gh-deploy --force
```

## Documentation Guidelines

### Style Guide

1. **Headers**: Use sentence case for headers
2. **Code blocks**: Always specify the language for syntax highlighting
3. **Links**: Use relative links for internal pages
4. **Examples**: Include practical, runnable examples
5. **Admonitions**: Use for important notes, warnings, and tips

### Markdown Extensions

Available extensions:

- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`
- **Code highlighting**: With line numbers and annotations
- **Tabbed content**: For alternative options
- **Tables**: Standard markdown tables
- **Emoji**: `:emoji_name:`

### Code Block Examples

#### With language specification:

```python
from finetune_cli import LLMFineTuner

finetuner = LLMFineTuner("gpt2")
```

#### With title:

```python title="example.py"
finetuner.load_model()
```

#### With line numbers:

```python linenums="1"
def train_model():
    finetuner.train()
```

### Admonition Examples

```markdown
!!! note
    This is a note admonition.

!!! warning
    This is a warning admonition.

!!! tip
    This is a tip admonition.

!!! example
    This is an example admonition.
```

## Contributing

To contribute to the documentation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

### Checklist

- [ ] Content is clear and concise
- [ ] Code examples are tested and work
- [ ] Links are valid
- [ ] Spelling and grammar are correct
- [ ] Follows existing documentation style
- [ ] Builds without errors locally

## Troubleshooting

### Build Errors

If you encounter build errors:

```bash
# Clean build cache
rm -rf site/

# Rebuild
mkdocs build --clean
```

### Missing Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
pip install mkdocs-material mkdocs-minify-plugin pymdown-extensions
```

### Preview Not Updating

Try clearing browser cache or use incognito mode.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## License

Documentation is licensed under MIT License, same as the project.