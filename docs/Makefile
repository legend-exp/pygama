SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)"

apidoc: clean-apidoc
	sphinx-apidoc --module-first --force --output-dir "$(SOURCEDIR)/generated" ../pygama $(PY_SOURCE_EXCLUDE)

clean-apidoc:
	rm -rf "$(SOURCEDIR)/generated"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
