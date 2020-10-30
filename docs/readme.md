### regenerating sphinx documentation
```
$ make clean
$ make html
$ open build/html/pygama.html
```
regenerate the .rst files:
```
$ sphinx-apidoc -f -o ./source ../pygama/
```