SHELL := /bin/bash
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

allver:
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIRALL)"
	branch=`cut -d. -f 1 <<< $(LATESTVER)`.`cut -d. -f 2 <<< $(LATESTVER)` && \
    if [ ! -f "$(BUILDDIRALL)/releases/$${branch}/index.html" ]; then \
        echo "ERROR: could not determine valid documentation for the latest version"; \
        exit 1; \
    else \
        content="\
<!DOCTYPE html>\n\
<html>\n\
  <head>\n\
    <title>Redirecting to $${branch} release</title>\n\
    <meta charset=\"utf-8\">\n\
    <meta http-equiv=\"refresh\" content=\"0; url=./releases/$${branch}/index.html\">\n\
    <link rel=\"canonical\" href=\"https://legend-exp.github.io/pygama/releases/$${branch}/index.html\">\n\
  </head>\n\
</html>"; \
        echo -e "$${content}" > "$(BUILDDIRALL)/index.html"; \
    fi

allver-info:
	sphinx-multiversion --dump-metadata "$(SOURCEDIR)" "$(BUILDDIRALL)"

apidoc: clean-apidoc
	sphinx-apidoc --private --module-first --force --output-dir "$(SOURCEDIR)/generated" ../src/pygama $(PY_SOURCE_EXCLUDE)

clean-apidoc:
	rm -rf "$(SOURCEDIR)/generated"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
