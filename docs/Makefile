SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build/html

all:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)"
	cp source/_assets/redirect.html "$(BUILDDIR)/index.html"

info:
	@$(SPHINXBUILD) --dump-metadata "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"
