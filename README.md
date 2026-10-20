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

three stages, in a loop:

1) histogram of bits using 4-wide radix width and 4-wide coarsening to get how many items have one of 16 bit combos across each block
2) single block, single stage hillis steele exclusive scan to find out how many positions each block will get for each index
3) scatter using digit and rank to put values and positions in their place

after each looper iteration, input and output are swapped, histograms are zeroed out, and we look at the next 4 of the 32 bits.

## Homework 5: TODO

## Homeowkr 6: TODO
