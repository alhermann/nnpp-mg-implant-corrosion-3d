# nnpp-mg-implant-corrosion-3d

C++ reference implementation to reproduce the **3D Nonlocal Nernst–Planck–Poisson (NNPP)** magnesium implant corrosion example from the associated publication.

- **Paper (SpringerLink):** https://link.springer.com/article/10.1007/s42102-024-00125-z  
- **Please cite:**  
  Hermann, Alexander, *et al.* “Nonlocal Nernst–Planck–Poisson system for modeling electrochemical corrosion in biodegradable magnesium implants.” *Journal of Peridynamics and Nonlocal Modeling* **7**(1) (2025): 1.

The code builds with **CMake**, uses **Eigen** (header-only) for linear algebra, **OpenMP** for parallel loops, and constructs peridynamic neighborhoods via **spatial hashing** (default) or an optional **kd-tree** (header-only, bundled).

---

## Features

- 3D NNPP corrosion example (research code).
- Neighborhood construction:
  - **Spatial hashing** — `src/helpers/grid.h` (default).
  - **kd-tree** (optional) — headers in `libraries/libkdtree++/kdtree++` with the adapter `src/helpers/kd_tree_interface.h`.
- Parallel loops with **OpenMP**.
- Pure C++11, single executable target.

---

## Repository layout

~~~
.
├─ CMakeLists.txt
├─ eigen/                          # Vendored Eigen headers (Eigen/…)
├─ src/
│  ├─ main.cpp
│  └─ helpers/
│     ├─ grid.h                    # Spatial hashing neighborhoods (default)
│     ├─ kd_tree_interface.h       # Optional kd-tree adapter
│     ├─ space.h
│     └─ utils.h
└─ libraries/
   └─ libkdtree++/
      └─ kdtree++/                 # Header-only kd-tree
~~~

> **Vendoring note:** You can place the **Eigen** header folder directly in the repo root as `./eigen/` so headers are at `eigen/Eigen/Dense`. Keep Eigen’s MPL-2.0 license file inside `eigen/`. Likewise, keep the **libkdtree++** license under `libraries/libkdtree++/`.

---

## Requirements

- **C++11** compiler (GCC, Clang, MSVC)
- **CMake** ≥ 3.0
- **OpenMP** (Linux/Windows: typically available; macOS: install `libomp` via Homebrew)

No system install of Eigen is required if you vendor `./eigen/` as shown above.

---

## Build (out-of-source)

~~~
# from the repository root
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j
~~~

> **Debug build:**
> ~~~
> cmake -DCMAKE_BUILD_TYPE=Debug ..
> cmake --build . -- -j
> ~~~

> **macOS (OpenMP):**
> ~~~
> brew install libomp
> # then configure & build as above
> ~~~

---

## Run

The executable expects **three arguments**:

~~~
./main <prefix> <jobname> <startfileNo>
~~~

- `<prefix>`: short label used in output naming/organization
- `<jobname>`: path to your case/input files (and where outputs may be grouped)
- `<startfileNo>`: (optional) integer index to start from (e.g., `0`)

**Example:**

~~~
# from build/ (where the executable lives after building)
./main -01 ../data/NNPP 0
~~~

If fewer than 3 arguments are provided, the program prints:

~~~
Usage: ./main <prefix> <jobname> <startfileNo>
~~~

---

## Switch neighborhood backend (optional)

- **Default:** Spatial hashing via `src/helpers/grid.h`.
- **kd-tree:** Headers in `libraries/libkdtree++/kdtree++` with an adapter at `src/helpers/kd_tree_interface.h`.

If your build needs a flag to toggle the kd-tree path, define it in your adapter (e.g., `#define USE_KDTREE 1`) or pass a compile definition at configure time:

~~~
cmake -DUSE_KDTREE=ON ..
~~~

(Ensure `kd_tree_interface.h` checks this macro.)

---

## Citation

If this repository helps your research, please cite:

> Hermann, Alexander, *et al.* “Nonlocal Nernst–Planck–Poisson system for modeling electrochemical corrosion in biodegradable magnesium implants.”  
> *Journal of Peridynamics and Nonlocal Modeling* **7**(1) (2025): 1.  
> https://link.springer.com/article/10.1007/s42102-024-00125-z

**BibTeX**

~~~bibtex
@article{Hermann2025NNPP,
  author  = {Hermann, Alexander and others},
  title   = {Nonlocal Nernst-Planck-Poisson system for modeling electrochemical corrosion in biodegradable magnesium implants},
  journal = {Journal of Peridynamics and Nonlocal Modeling},
  volume  = {7},
  number  = {1},
  pages   = {1},
  year    = {2025},
  doi     = {10.1007/s42102-024-00125-z},
  url     = {https://link.springer.com/article/10.1007/s42102-024-00125-z}
}
~~~

---

## License & third-party

- **This repository:** MIT License (see `LICENSE`)
- **Third-party (vendored):**
  - **Eigen** — MPL-2.0. Keep the MPL-2.0 license text in `eigen/`.
  - **libkdtree++** — Artistic License 2.0. Keep its license under `libraries/libkdtree++/`.

See `THIRD_PARTY_NOTICES.md` for details and links.

---

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue first to discuss what you’d like to change.
