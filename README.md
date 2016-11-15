# ConvexSwitch

[![Build Status](https://travis-ci.org/YeeJeremy/ConvexSwitch.jl.svg?branch=master)](https://travis-ci.org/YeeJeremy/ConvexSwitch.jl)

[![Coverage Status](https://coveralls.io/repos/YeeJeremy/ConvexSwitch.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/YeeJeremy/ConvexSwitch.jl?branch=master)

[![codecov.io](http://codecov.io/github/YeeJeremy/ConvexSwitch.jl/coverage.svg?branch=master)](http://codecov.io/github/YeeJeremy/ConvexSwitch.jl?branch=master)

## Authors

Juri Hinz and Jeremy Yee

## About

This Julia package provides a method in which convex switching systems
can be numerically solved. A work in progress. The primal-dual methods are
implemented by return incorrect results. The Bellman recursion methods
seem to be fine though.

BUG: Bounds are not tight. Either martingale increments are not computed
correctly or the duality recutsion is buggy.
