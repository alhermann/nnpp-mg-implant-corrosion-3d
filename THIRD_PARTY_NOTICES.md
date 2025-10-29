# Third-Party Notices

This project vendors third-party, header-only libraries. Their original license terms and notices are preserved below and in the referenced files/directories. If you redistribute this repository (in source or binary form), **keep these notices and license texts**.

---

## 1) Eigen

- **What:** C++ template library for linear algebra (matrices, vectors, numerical solvers).  
- **Upstream:** https://eigen.tuxfamily.org  
- **License:** **Mozilla Public License 2.0 (MPL-2.0)**  
- **SPDX:** `MPL-2.0`  
- **Included at:** `./eigen/` (headers such as `eigen/Eigen/Dense`, `eigen/Eigen/Sparse`, etc.)  
- **License text in repo:** `eigen/COPYING.MPL2` (or `eigen/LICENSE*` as provided upstream)

**Notice & obligations (summary, not legal advice):**
- Eigen is shipped here **unmodified** as header-only files.  
- MPL-2.0 is a *file-level* copyleft: if you modify **Eigen’s files**, your modified versions of those files must remain under MPL-2.0 and retain the license header/notice.  
- Your **own** project files (that simply include Eigen headers) may remain under your chosen license (e.g., MIT), provided you respect the MPL terms.

**Attribution:**  
© The Eigen authors and contributors. All rights reserved under MPL-2.0.

---

## 2) libkdtree++

- **What:** Header-only kd-tree implementation for nearest-neighbor queries.  
- **Canonical/mirror:** https://github.com/nvmd/libkdtree (commonly used mirror)  
- **License:** **Artistic License 2.0**  
- **SPDX:** `Artistic-2.0`  
- **Included at:** `./libraries/libkdtree++/kdtree++/`  
- **License text in repo:** `libraries/libkdtree++/LICENSE` (or as supplied in the library directory)

**Notice & obligations (summary, not legal advice):**
- The library is included **unmodified**.  
- You must retain the original license text and notices. Redistribution and modification are permitted under the Artistic 2.0 terms.

**Attribution:**  
© The libkdtree++ authors and contributors. All rights reserved under Artistic License 2.0.

---

## How these are used

- **Header-only usage:** Both libraries are consumed as headers during compilation; no separate linking steps are required.  
- **Include paths (CMake):**
  - Eigen: `eigen/` (so headers reside at `eigen/Eigen/...`)  
  - libkdtree++: `libraries/libkdtree++/` (headers under `kdtree++/`)

---

## If you update or modify vendored code

1. **Do not remove license files or notices.**  
2. **Eigen (MPL-2.0):** Keep modified Eigen files under MPL-2.0 with proper headers; your own project files can remain under your chosen license.  
3. **libkdtree++ (Artistic-2.0):** Keep the license file and notices; follow the Artistic 2.0 terms for modifications and redistribution.  
4. Consider documenting the upstream version/commit and date in this file or within the corresponding directory.

---

## Project license

The **project’s own** source code is licensed under **MIT** (see `LICENSE`).  
This does **not** supersede third-party license obligations for files under `eigen/` and `libraries/libkdtree++/kdtree++/`.

---

*This file is provided for clarity and convenience only and is not legal advice. Refer to the full license texts for binding terms.*
