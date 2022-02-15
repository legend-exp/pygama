SPHINXBUILD   ?= sphinx-build
SPHINXMULTI   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIRALL   = build/allver/html
BUILDDIR      = build

all: this
	@$(SPHINXMULTI) "$(SOURCEDIR)" "$(BUILDDIRALL)"
	cp source/_assets/redirect.html "$(BUILDDIRALL)/index.html"

this:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

info:
	@$(SPHINXMULTI) --dump-metadata "$(SOURCEDIR)" "$(BUILDDIRALL)"

clean:
	rm -rf "$(BUILDDIR)"
