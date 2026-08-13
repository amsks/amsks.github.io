# amsks.github.io

Source for my personal academic site: [amsks.github.io](https://amsks.github.io).

Built with [al-folio](https://github.com/alshedivat/al-folio) (MIT, see `LICENSE`)
on Jekyll, deployed to GitHub Pages by `.github/workflows/deploy.yml` on every
push to `master`.

## Local development

ImageMagick is a hard requirement, not an optional extra. The `jekyll-imagemagick`
plugin shells out to `convert` to generate the responsive `.webp` variants that
`_includes/figure.html` references in its `srcset`. Without it the build still
reports success, but every image tag points at a file that was never written.

```bash
brew install imagemagick     # or: apt-get install imagemagick
bundle install
bundle exec jekyll serve
```

The site is then at <http://127.0.0.1:4000>.

Ruby version is pinned in `.ruby-version` and gem versions in `Gemfile.lock`;
CI reads both, so a local build and a deploy resolve the same dependencies.

## Where things live

| Path | Contents |
|---|---|
| `_pages/` | About, CV, and publications pages |
| `_posts/` | Blog posts, `YYYY-MM-DD-kebab-slug.md` |
| `_news/` | Homepage news items, same naming convention |
| `_bibliography/papers.bib` | Publication list, rendered by jekyll-scholar |
| `_data/cv.yml` | Structured CV driving the `/cv/` page |
| `assets/pdf/CV_Academic_latest.pdf` | PDF CV linked from `/cv/` |

Adding a news item means dropping a file in `_news/` with `layout`, `title`,
`date`, and `inline: true`. The date in the front matter is what orders and
labels the entry; the date in the filename is there to keep the directory
readable and is not parsed.

Publication years are listed explicitly in `_pages/publications.md` under
`years:`. A new year has to be added there or its entries will not render.

## Formatting

`pre-commit` handles trailing whitespace, end-of-file newlines, and YAML
validity:

```bash
pre-commit install
```
