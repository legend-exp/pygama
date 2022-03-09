### Documentation
```
$ git fetch origin
$ make clean
$ make
$ open build/html/pygama.html
```
regenerate the `.rst` files:
```
$ sphinx-apidoc -f -o ./source ../pygama/
