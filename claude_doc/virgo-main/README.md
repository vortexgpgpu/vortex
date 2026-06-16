Virgo
=====

Virgo is a GPU microarchitecture that integrates dedicated matrix units at the
cluster (SM)-level, achieving better FLOPS scalability and energy efficiency.

<p align="center">
  <img src="https://raw.githubusercontent.com/ucb-bar/virgo/refs/heads/main/img/fig-cluster-overview.svg" alt="Virgo cluster microarchitecture overview" width="600">
</p>

This repository includes the essential RTL logic for Virgo's implementation,
including the Gemmini matrix unit integration, shared memory, baseline Tensor
Core models, memory coalescer, and Vortex SIMT core integration.

The entire Virgo GPU design is implemented within the Chipyard SoC environment.
To evaluate the full design, please follow the instructions in
[Chipyard](https://github.com/ucb-bar/chipyard/commits/virgo/).

The GPU kernel written and evaluated for Virgo can be found in
[virgo-kernels](https://github.com/ucb-bar/virgo-kernels).


Code Structure
--------------

A Virgo cluster is constructed by integrating a collection of `Tile`s that
house Virgo's compute units, such as Vortex SIMT cores and the Gemmini matrix
unit, as well as memory units such as the shared memory and interconnect.  We
use rocket-chip's [Cluster
API](https://github.com/chipsalliance/rocket-chip/blob/master/src/main/scala/subsystem/Cluster.scala)
to construct the cluster hiearchy.

The Chisel RTL code for the main Virgo hardware modules can be found in
`src/main/scala/radiance`:

* [`tile`](src/main/scala/radiance/tile)
  * [`RadianceCluster.scala`](src/main/scala/radiance/tile/RadianceCluster.scala): Top-level definition of a Virgo Cluster
  * [`RadianceSharedMem.scala`](src/main/scala/radiance/tile/RadianceSharedMem.scala): **Shared memory and interconnect** implementation
  * [`RadianceTile.scala`](src/main/scala/radiance/tile/RadianceTile.scala): Vortex SIMT core tile
  * [`GemminiTile.scala`](src/main/scala/radiance/tile/GemminiTile.scala): Gemmini-based **Virgo matrix unit and MMIO** instantiation
  * [`VortexCore.scala`](src/main/scala/radiance/tile/VortexCore.scala): Chisel wrapper module for **Vortex SIMT core**
  * [`Barrier.scala`](src/main/scala/radiance/tile/Barrier.scala): Cluster-wide **barrier synchronizer** module
* [`memory`](src/main/scala/radiance/memory)
  * [`Coalescing.scala`](src/main/scala/radiance/memory/Coalescing.scala): **Memory coalescer** implementation
  * [`SyncMem.scala`](src/main/scala/radiance/memory/SyncMem.scala): SRAM implementation for the shared memory
  * `*Node.scala`: Arbiter and multiplexer nodes used in the shared memory interconnect
* [`core`](src/main/scala/radiance/core)
  * [`TensorCoreDecoupled.scala`](src/main/scala/radiance/core/TensorCoreDecoupled.scala):
    **Hopper-style Tensor Core** implementation
  * [`TensorDPU.scala`](src/main/scala/radiance/core/TensorDPU.scala):
    Four-element **dot-product units** used in Tensor Core implementations
* [`subsystem`](src/main/scala/radiance/subsystem): Chipyard Config definitions for parameterizing clusters

More details to follow.

Publications
------------

[Virgo: Cluster-level Matrix Unit Integration in GPUs for Scalability and Energy Efficiency (ASPLOS'25)](https://dl.acm.org/doi/abs/10.1145/3676641.3716281)

```
@inproceedings{kim2025virgo,
  title={Virgo: Cluster-level Matrix Unit Integration in GPUs for Scalability and Energy Efficiency},
  author={Kim, Hansung and Yan, Ruohan Richard and You, Joshua and Yang, Tieliang Vamber and Shao, Yakun Sophia},
  booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  pages={1382--1399},
  year={2025}
}
```
