# Finite-Element-Method-in-Structural-Dynamics

# FEM Structural Dynamics: Coal Conveyor System Analysis

[cite_start]This project develops a comprehensive Finite Element Method (FEM) model to analyze the static and dynamic behavior of a steel beam structure inspired by a coal hauling conveyor system[cite: 23]. [cite_start]The analysis, performed at Politecnico di Milano [cite: 3, 4][cite_start], focused on determining the dynamic properties, computing responses to various loads, and optimizing the structure[cite: 24, 646].

---

## Methodologies and Analysis

[cite_start]The structure consists of interconnected steel beams (IPE 220, IPE 100, IPE 160) distributed along an inclined geometry[cite: 44].

* [cite_start]**FE Modeling:** A linear beam element model was developed to ensure **quasi-static behavior** up to $20~Hz$[cite: 30, 59]. [cite_start]Element lengths were selected based on beam properties and maximum frequency to maintain accuracy[cite: 61, 62].
* [cite_start]**Modal Analysis:** The inherent dynamic properties were determined by solving the generalized eigenvalue problem[cite: 135].
    * [cite_start]First three natural frequencies: **6.71 Hz**, **8.22 Hz**, and **14.36 Hz**[cite: 199].
* [cite_start]**Frequency Response Function (FRF):** FRFs were calculated using both the full FEM approach and **modal superposition** (using the first two modes)[cite: 34, 203]. [cite_start]This confirmed the modal method's accuracy in the low-frequency and quasi-static regions[cite: 637].
* [cite_start]**Static Response:** The static response to gravitational load was analyzed[cite: 36, 347]. [cite_start]The maximum vertical deflection at Point A was initially **6.69 mm**[cite: 379].
* [cite_start]**Time-Domain Response:** The dynamic response to a concentrated load moving at $2~m/s$ was simulated [cite: 38, 384][cite_start], highlighting the structure's transient vibration and the **snap-back effect** upon load exit[cite: 519, 642].

---

## Optimization Results

[cite_start]The final design successfully reduced the maximum static deflection at the free end by over 50% by strategically stiffening the V-shaped support[cite: 644, 645].

| Metric | Initial Deflection (mm) | Final Deflection (mm) | Change (%) | Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Max Deflection** | -6.69 | [cite_start]-2.93 [cite: 627] | [cite_start]**-56.20%** [cite: 627] | [cite_start]$> 50\%$ [cite: 609] |
| **Mass Increase** | N/A | N/A | [cite_start]**+8.38%** [cite: 627] | [cite_start]$< 20\%$ [cite: 609] |

---

## 💻 Tools Employed

* [cite_start]**MATLAB:** Primary platform for all FE modeling, matrix assembly, modal analysis, time integration (using `ode45` solver [cite: 460]), and result visualization.
* [cite_start]**AutoCAD:** Used for initial geometric definition and node coordinate extraction[cite: 121, 122].

---

## 🎯 Intended Audience

[cite_start]This repository is a practical reference for **Students of Structural Dynamics and FEM** [cite: 5, 6][cite_start], demonstrating the complete workflow from model discretization and modal analysis to complex time-domain simulation and structural optimization based on performance criteria[cite: 646].
