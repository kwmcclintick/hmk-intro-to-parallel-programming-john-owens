# hmk-intro-to-parallel-programming-john-owens
homework assignements i've completed following along with the youtube course

## Hello World

the squaring program

## Homework 1: Greyscale

one in, one out mapping to/from global VRAM, one per thread

## Homework 2: Blur

stencil (von neumann 2d "cross"). shared memory for memory re-use and careful bounds checking to 1) read from shared if you can 2) read from VRAM if you can't 3) don't read anything if it's out of bounds. Per-color sums tracked and divided by count back to VRAM

## Homework 3: tone mapping

three stages: 1) two-kernel min/max binary tree reduce using shared block SRAM, coarsening for latency hiding, and warp register sharing for reduced syncs. Then, 2) two-kernel histogram with coarsening, into a reduce, then 3) an exclusive hillis steele scan on the histogram to form a CDF

## Homework 4: red eye removal

work in progress

## Homework 5: TODO

## Homeowkr 6: TODO
