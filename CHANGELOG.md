# Changelog

## [1.0.1](https://github.com/infer-actively/pymdp/compare/v1.0.0...v1.0.1) (2026-04-28)


### Bug Fixes

* **maths:** clip y in stable_cross_entropy to prevent NaN/-inf ([5db35bb](https://github.com/infer-actively/pymdp/commit/5db35bb1bd3b5f07b18735092b719becbfd11b27))
* **maths:** clip y in stable_cross_entropy to prevent NaN/-inf ([9709f91](https://github.com/infer-actively/pymdp/commit/9709f919ac6814e537c6d337180e5ed26f045908)), closes [#394](https://github.com/infer-actively/pymdp/issues/394)
* replace mutable default `shape=[2,2]` with `None` guard in grid_worlds.py ([0c7db01](https://github.com/infer-actively/pymdp/commit/0c7db01f41cecc40ed2e8d5c46db88bc4989ae15))
* replace mutable default `shape=[2,2]` with `None` guard in grid_worlds.py ([f1df9ed](https://github.com/infer-actively/pymdp/commit/f1df9ed5517f57069fdc27529816fc60d51a4f96)), closes [#392](https://github.com/infer-actively/pymdp/issues/392)


### Dependencies

* mirror jax/jaxlib &lt;0.10 pin in setup.cfg ([47604ec](https://github.com/infer-actively/pymdp/commit/47604ec41e20dbcb59cae7704a76f8412c8d849e))
* pin jax/jaxlib &lt;0.10 pending numpyro fix ([0c2a821](https://github.com/infer-actively/pymdp/commit/0c2a821d4a6b107a310db279119d162c849f0675))


### Documentation

* make examples notebooks the single source of truth ([3c58d8f](https://github.com/infer-actively/pymdp/commit/3c58d8f173dcd03d98e49925bf5bede5588b11e5))
* **notebook:** fix center_left preference comment in model construction tutorial ([ef8824e](https://github.com/infer-actively/pymdp/commit/ef8824e6f0732ad9fbce3445827dad131ed6f46f))
* **notebook:** fix center_left preference comment in model construction tutorial ([9050c02](https://github.com/infer-actively/pymdp/commit/9050c02e33948bc18726acf110ab16c58359a1e7))
* refresh PyPI badge for 1.0.0 [skip ci] ([1c8e99f](https://github.com/infer-actively/pymdp/commit/1c8e99f46a8a2bddb9395fd865667794393c2421))
