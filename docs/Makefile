SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build/html

all:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)"
	cp source/_assets/redirect.html "$(BUILDDIR)/index.html"

clean:
	rm -rf "$(BUILDDIR)"
