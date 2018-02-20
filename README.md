# Scene understading for autonomous vehicles
This project is done in the framework of the Master in Computer Vision by
Universitat Autònoma de Barcelona (UAB) in the module _M5: Visual recognition_.

<!-- Short abstract (2-lines) -->

For more information check our report in [Overleaf][overleaf] or [Google Drive][gdrive]

## Getting Started

### Prerequisites
The software need the following software to run:

- [Python 2.7][python27]
- [pip][pip-pypi]
- [Pipenv][pipenv-docs]

If you are using any Linux distribution is most probable you already have a
running version of Python installed.

As an example, if you are using an Ubuntu you can install pip and pipenv using
these commands:

```sh
apt-get install python-pip
pip install pipenv
```


### Installing
To install the virtual environment you only have to run pipenv from project's
root directory:

```sh
pipenv install
```

It is recommended to add the following environment vars to your `.bashrc`:

```sh
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
export PIPENV_MAX_DEPTH=1
```

You can find their explanation as well as more environment variables in
[Pipenv's documentation][pipenv-env-vars]


## Built With
This project is using [Keras][keras] as a high-level neural networks API running
on top of [Tensorflow][tf] library.


## Authors
- Ferran Pérez              - _dev_ - [fperezgamonal][ferran-github] - [contact](mailto:ferran.perezg@e-campus.uab.cat)
- Joan Francesc Serracant   - _dev_ - [fserracan][cesc-github] -  [contact](mailto:joanfrancesc.serracant@e-campus.uab.cat)
- Jonatan Poveda            - _dev_ - [jonpoveda][jonatan-github] - [contact](mailto:jonatan.poveda@e-campus.uab.cat)
- Martí Cobos               - _dev_ - [marticobos][marti-github] - [contact](mailto:marti.cobos@e-campus.uab.cat)


## License
This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

<!--
## Acknowledgements
-->

[python27]: https://docs.python.org/2/
[pip-pypi]: https://pypi.python.org/pypi/pip
[pipenv-docs]: http://pipenv.readthedocs.io/en/latest/
[pipenv-env-vars]: http://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables
[keras]: https://keras.io
[tf]: https://www.tensorflow.org

[ferran-github]: https://github.com/fperezgamonal
[cesc-github]: https://github.com/fserracant
[jonatan-github]: https://github.com/jonpoveda
[marti-github]: https://github.com/marticobos

[overleaf]: https://www.overleaf.com/read/rgbqdstbtmqz
[gdrive]: https://docs.google.com/presentation/d/1fmX2s14--DSvh6eTJD6e-rf5zkyVoq6O_012BAI6jJs/edit?usp=sharing
