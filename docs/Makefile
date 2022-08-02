SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

# list in source/.py_api_exclude files or directories that should be excluded
# from the API auto-generated documentation. No regex allowed
PY_SOURCE_EXCLUDE = $(shell cat source/.py_api_exclude)

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

apidoc: clean-apidoc
	sphinx-apidoc --private --module-first --force \
      --output-dir "$(SOURCEDIR)/generated" \
      ../src/pygama $(PY_SOURCE_EXCLUDE)

clean-apidoc:
	rm -rf "$(SOURCEDIR)/generated"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
