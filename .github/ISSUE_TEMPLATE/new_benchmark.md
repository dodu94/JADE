---
name: New Benchmark
about: Describe a new benchmark to be added to the project
title: "[New Benchmark] - title of the issue"
labels: new benchmark
assignees: ''

---
# To Be completed for all benchmarks

## Is the new benchmark computational or experimental?

- [ ] Computational
- [ ] Experimental

## Is this a MultiTest or a normal Test?
Specify here if the benchmark consists of only one transport run or it is a collection of input files that have the same general structure.

## Give a concise description of the benchmark
Give here a concise description of the benchmark. Include links to web pages where a longer description of the benchmark is available. Attach description documents of the
benchmark to this issue if available.

## Does the benchmark needs new types of plots?
You can check the available plot types already implemented in JADE [here](https://jade-a-nuclear-data-libraries-vv-tool.readthedocs.io/en/latest/usage/postprocessing.html#plots-atlas). If you think that no plot can be adapted to the needs of this benchmark please provide a description (better a screenshot) of the new type of plots
to be implemented.

## Can the benchmark be used for all codes implemented in JADE or for a specific subset?
Specify here for which codes the benchmark should be developed.

## Can inputs of this benchmark be freely re-distributed?
- [ ] yes
- [ ] no

If no, please specificy who will be able to access the inputs.

## Additional information/context
Add here any additional information or context that you want to provide.

# To be completed only for experimental benchmarks

Please erase this entire block if the benchmark to be added is a computational one.

## Is any of the existing general classes suitable for this benchmark?
Some generic classes have been already developed for experimental classes of benchmarks that can be customized through excel config file. Have a look at the
[related documentation section](https://jade-a-nuclear-data-libraries-vv-tool.readthedocs.io/en/latest/dev/insertbenchmarks.html#insert-custom-experimental-benchmark) and specify here if the new benchmark
can use one of these classes or a new one needs to be created.

## Can experimental data of this benchmark be freely re-distributed?
- [ ] yes
- [ ] no

If no, please specificy who will be able to access the experimental data.