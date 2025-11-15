# Finite-Element-Method-in-Structural-Dynamics

# FEM Analysis of a Steel Conveyor Structure: Dynamic and Static Behavior

This project, undertaken as part of the Advanced Dynamics of Mechanical Systems course, involved the development of a comprehensive **Finite Element (FE) model** to investigate the structural dynamics of a mechanical structure inspired by a coal hauling conveyor system. The objective was to develop and implement the FE model to analyze the structural response under various load conditions, ultimately leading to a design optimization.

---

## Key Methodologies and Analysis

The analysis involved a systematic approach to characterize the structure's dynamic and static properties:

* **FE Modeling and Discretization:** The structure, composed of interconnected steel beams (IPE 220, IPE 100, and IPE 160 profiles), was discretized into elemental components. The modeling considered a frequency range up to $20~Hz$, requiring a precise calculation of the maximum allowable element length to ensure **quasi-static behavior**.
* **Modal Analysis:** This fundamental technique was used to determine the inherent dynamic properties of the structure: its natural frequencies and corresponding mode shapes.
* **Frequency Response Functions (FRF):** FRFs were calculated both via the direct FEM approach and through the **modal superposition approach**. The comparison highlighted the modal approach's accuracy in the low-frequency and quasi-static regions, while underscoring its limitations at higher frequencies where additional modes become crucial.
* **Static and Time-Domain Response:** The static response due to gravitational load was analyzed, identifying critical deflection areas. The time history response was evaluated under the action of a **moving load** traveling at a constant velocity, simulating a realistic loading scenario.

---

## Project Conclusions and Optimization

The project successfully applied FEM in structural dynamics, offering valuable insights for performance prediction and design improvement:

* **Dynamic Response Insights:** The moving load analysis enabled a realistic simulation, highlighting the induced vibrations and confirming the presence of the **snap-back effect** once the load exited the structure. This demonstrated the importance of dynamic modeling in capturing transient effects.
* **Structural Optimization:** A primary goal was achieved by proposing a structural modification that reduced the maximum static deflection by over **50%** with an acceptable increase in mass. This was accomplished by strategically stiffening the structure's V-shaped support, demonstrating the effectiveness of targeted structural modifications based on a thorough understanding of the system’s behavior.

In summary, this assignment showcased the comprehensive application of FEM for performance prediction and design optimization in complex structural systems.

## Tools Employed

* **MATLAB:** Used for all FE modeling, matrix assembly, modal analysis, time integration (solving the second-order differential equation using the `ode45` solver), and result visualization.
* **AutoCAD:** Used for initial geometric definition and node coordinate extraction to define the input file.

---

## Intended Audience

This repository is a practical reference for **Students of Structural Dynamics and FEM**, demonstrating the complete workflow from model discretization and modal analysis to complex time-domain simulation and structural optimization based on performance criteria. It is also useful for engineers interested in **Finite Element Modeling of Beam Structures** and **Modal Superposition** methods.
