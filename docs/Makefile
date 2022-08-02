SOURCEDIR = source
BUILDDIR = build
LATESTVER = $(shell git describe --tags --abbrev=0)

# docs for all project versions will be placed here
BUILDDIRALL = $(BUILDDIR)/allver/html

# list in source/.py_api_exclude files or directories that should be excluded
# from the API auto-generated documentation. No regex allowed
PY_SOURCE_EXCLUDE = $(shell cat source/.py_api_exclude)

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

allver: update-redirect
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIRALL)"

allver-info:
	sphinx-multiversion --dump-metadata "$(SOURCEDIR)" "$(BUILDDIRALL)"

apidoc: clean-apidoc
	sphinx-apidoc --private --module-first --force --output-dir "$(SOURCEDIR)/generated" ../src/pygama $(PY_SOURCE_EXCLUDE)

clean-apidoc:
	rm -rf "$(SOURCEDIR)/generated"

update-redirect:
	branch=`cut -d. -f 1 <<< $(LATESTVER)`.`cut -d. -f 2 <<< $(LATESTVER)` && \
    ln -fs "releases/$${branch}/index.html" "$(BUILDDIRALL)/index.html"
	@if [ ! -f `readlink -f "$(BUILDDIRALL)/index.html"` ]; then \
        echo "ERROR: could not determine valid documentation for the latest version"; \
        exit 1; \
    fi

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
