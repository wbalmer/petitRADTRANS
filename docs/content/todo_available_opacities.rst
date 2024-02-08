.. _avail_opas:

TODO: Available opacity species
===============================

Line species
____________

The line opacities that can be downloaded `via Keeper <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0>`_ in petitRADTRANS are listed below.

To add more opacities, please see `Adding opacities <opa_add.html>`_, among them how to plug-and-play install the Exomol opacities calculated in the pRT format, available from the `Exomol website <http://www.exomol.com/data/data-types/opacity/>`_.

.. important::
   Please cite the reference mentioned in the description (click the link) when making use of a line species listed below. Information about the opacity source are also available in the opacity HDF file under the key ``DOI`` and its attributes.

File naming convention
^^^^^^^^^^^^^^^^^^^^^^

In petitRADTRANS, species follow a naming convention similar to that of `ExoMol <https://www.exomol.com/>`_. It is indicated below.

- Species names are based on their chemical formula.
- Elements in the chemical formula are separated by ``-``.
- The number in front of the element indicates its isotope, when relevant.
- The number after the element indicates its quantity in the molecule, when relevant.
- Opacities considering the natural (i.e. Earth) abundance of isotopologue are indicated with the string ``-NatAbund`` after the chemical formula.
- The charge of the species is indicated after the formula, starting with ``_``. The character ``p`` is used for positive charges and ``n`` for negative charges.
- The number in front of the charge indicates the charge amount.
- The source of the opacity is indicated after the charge, starting with ``__``.
- The spectral information of the opacity is indicated after the source, starting with ``.``.
- The character ``R`` indicates constant resolving power (:math:`\lambda/\Delta\lambda` constant).
- The string ``DeltaWavenumber`` indicates constant spacing in wavenumber (:math:`\Delta\nu` constant).
- The string ``DeltaWavelength`` indicates constant spacing in wavelength (:math:`\Delta\lambda` constant).
- The number coming after the above indicates the spacing.
- The wavelength range, in µm, is indicated afterward, starting with a ``_`` and ending with ``mu``. The upper and lower boundaries are separated with ``-``.

.. important::
	1. In petitRADTRANS, writing the full opacity name in a script is often not necessary.
   	2. The ``line_species`` opacity name and the ``mass_fractions`` dictionary keys must match *exactly*.

"Low-resolution" opacities (``"c-k"``, :math:`\lambda/\Delta\lambda=1000`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In correlated-k mode (``"c-k"``), most of the molecular opacities are calculated considering only the main isotopologue. Most of the time, the differences with including all isotopologues, at these resolving powers, are negligible (see comparison with `Baudino et al., 2017 <https://www.doi.org/10.3847/1538-4357/aa95be>`_).

For some species such as CO and TiO, the contribution of all isotopologues is considered, following their natural abundance **on Earth**. Some secondary isotopologues are also available. This has been done because of a large natural abundance ratio between the isotopes of some elements (e.g. Ti), and/or because of the significant spectral contribution of secondary isotopologues at the considered resolution (e.g. 12CO/13CO).

All ``c-k`` opacities have a resolving power of 1000 and cover **at least** wavelengths 0.3 to 50 µm. Pressure and temperature grids may vary. All of the opacities are sampled over 16 k-coefficients following the method described in `Baudino et al. (2015) <https://doi.org/10.1051/0004-6361/201526332>`_

The available correlated-k opacities are listed below. When multiple source are available for species, the recommended one is indicated in bold.

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - Species name
     - Short file name*
     - Reference
     - Contributor
   * - Al
     - 27Al__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Al+
     - 27Al_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - AlH
     - 27Al-1H__AlHambra
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+18 <https://doi.org/10.1093/mnras/sty1524>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - AlO
     - 27Al-16O__ATP
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Patrascu+15 <http://dx.doi.org/10.1093/mnras/stv507>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - C2H2
     - 12C2-1H2__aCeTY
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Chubb+20 <https://doi.org/10.1093/mnras/staa229>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - C2H4
     - 12C2-1H4__MaYTY
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Mant+18 <https://doi.org/10.1093/mnras/sty1239>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - Ca+
     - 40Ca_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - CaH
     - 40Ca-1H
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Li+12 <http://dx.doi.org/10.1016/j.jqsrt.2011.09.010>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - CH3D
     - 12C-1H3-2H__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - CH4
     - 12C-1H4__HITEMP
     - **???**
     - **???**
   * - **CH4**
     - 12C-1H4__YT34
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+17 <https://doi.org/10.1051/0004-6361/201731026>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - 13CH3D
     - 13C-1H3-2H__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - 13CH4
     - 13C-1H4__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - CO
     - C-O-NatAbund__HITEMP
     - All isotopologues, HITEMP/Kurucz, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - **CO**
     - C-O-NatAbund__Chubb
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - 12CO
     - 12C-16O__HITEMP
     - HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - 13CO
     - 13C-16__HITEMP
     - HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - **13CO**
     - 13C-16O__Li2015
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - CO2
     - 12C-16O2__UCL
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_
     - --
   * - CrH
     - 52Cr-1H__MoLLIST
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Burrows+02 <http://dx.doi.org/10.1086/342242>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - CS2
     - C-S2-NatAbund__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - Fe
     - 56Fe__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Fe+
     - 56Fe_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - FeH
     - 56Fe-1H__MoLLIST
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Wende+10 <http://dx.doi.org/10.1051/0004-6361/201015220>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - H2
     - 1H2__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - H2O
     - 1H2-16O__HITEMP
     - HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - **H2O**
     - 1H2-16O__POKAZATEL
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Polyanski+18 <https://doi.org/10.1093/mnras/sty1877>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - H2-17O
     - 1H2-17O__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - H2-18O
     - 1H2-18O__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - HDO
     - 1H-2H-16O__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - H2S
     - 1H2-32S__AYT2
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Azzam+16 <http://dx.doi.org/10.1093/mnras/stw1133>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - HCN
     - 1H-12C-14N__Harris
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Barber+14 <http://mnras.oxfordjournals.org/content/437/2/1828.abstract>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - **K**
     - 39K_Allard
     - VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - K
     - 39K__Burrows
     - VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
     - --
   * - K
     - 39K_LorCut
     - VALD, Lorentzian wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - Li
     - 3Li__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Mg
     - Mg__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Mg+
     - 24Mg_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - MgH
     - 24Mg-1H__MoLLIST
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gharib-Nezhad+13 <http://dx.doi.org/10.1093/mnras/stt510>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - MgO
     - 24Mg-16O__LiTY
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Tennyson+19 <https://doi.org/10.1093/mnras/stz912>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - **Na**
     - 23Na_Allard
     - VALD, `new Allard wings <https://ui.adsabs.harvard.edu/abs/2019yCat..36280120A/abstract>`_, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - Na
     - 23Na__Burrows
     - Main isotopologue, VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
     - --
   * - Na
     - 23Na_LorCut
     - VALD, Lorentzian wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - NaH
     - 23Na-1H
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Rivlin+15 <http://dx.doi.org/10.1093/mnras/stv979>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - NH3
     - 14N-1H3__CoYuTe
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Coles+19 <https://doi.org/10.1093/mnras/stz2778>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - 15NH3
     - 15N-1H3__HITRAN
     - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
     - --
   * - O
     - 16O__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - O2
     - 16O2__HITRAN
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+17 <https://doi.org/10.1016/j.jqsrt.2017.06.038>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - O16-O17
     - 16O-17O__HITRAN
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+17 <https://doi.org/10.1016/j.jqsrt.2017.06.038>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - O16-O18
     - 16O-18O__HITRAN
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+17 <https://doi.org/10.1016/j.jqsrt.2017.06.038>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - O3
     - 16O3__HITRAN
     - HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - OH
     - 16O-1H__MoLLIST
     - M`ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Brooke+16 <http://dx.doi.org/10.1016/j.jqsrt.2015.07.021>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - PH3
     - 31P-1H3__SAlTY
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Sousa-Silva+14 <http://dx.doi.org/10.1093/mnras/stu2246>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - SH
     - 32S-1H__GYT
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gorman+19 <https://doi.org/10.1093/mnras/stz2517>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - Si
     - 28Si__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Si+
     - 28Si_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - SiO
     - 28Si-16O__SiOUVenIR
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Barton+13 <https://doi.org/10.1093/mnras/stt1105>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - SiO2
     - 28Si-16O2__OYT3
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Owens+20 <http://dx.doi.org/10.1093/mnras/staa1287>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - SO2
     - 32Si-16O2__ExoAmes
     - **???**
     - **???**
   * - Ti
     - 48Ti__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Ti+
     - 48Ti_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - **TiO**
     - Ti-O__McKemmish
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+19 <https://doi.org/10.1093/mnras/stz1818>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - TiO
     - Ti-O-NatAbund_Plez
     - B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - 48TiO
     - 48Ti-16O__Plez
     - B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - **48TiO**
     - 48Ti-16O__McKemmish
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+19 <https://doi.org/10.1093/mnras/stz1818>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_
   * - V
     - 51V__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - V+
     - V_p__Kurucz
     - `Kurucz <http://kurucz.harvard.edu>`_
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - VO
     - 51V-16O__Plez
     - B. Plez,, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - **VO**
     - 51V-16O__VOMYT
     - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+16 <http://dx.doi.org/10.1093/mnras/stw1969>`_
     - `K. Chubb <klc20@st-andrews.ac.uk>`_

*: discarding the spectral information.


**Line absorbers, high resolution mode** (``"lbl"``, with :math:`\lambda/\Delta\lambda=10^6`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary*
     - Description
     - Contributor
   * - C2H2_main_iso
     - C2H2_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CH4_212
     - CH4_212
     - :math:`\rm CH_3D`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CH4_Hargreaves_main_iso
     - CH4_Hargreaves_main_iso
     - Main isotopologue, HITEMP, see `Hargreaves et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJS..247...55H/abstract>`_
     - --
   * - CO2_main_iso
     - CO2_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_27
     - CO_27
     - :math:`\rm ^{12}C^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_28
     - CO_28
     - :math:`\rm ^{12}C^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_36
     - CO_36
     - :math:`\rm ^{13}C^{16}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_37
     - CO_37
     - :math:`\rm ^{13}C^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_38
     - CO_38
     - :math:`\rm ^{13}C^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_all_iso
     - CO_all_iso
     - All isotopologues, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - CO_main_iso
     - CO_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_162
     - H2O_162
     - :math:`\rm HDO`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_171
     - H2O_171
     - :math:`\rm H_2 \ ^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_172
     - H2O_172
     - :math:`\rm HD^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_181
     - H2O_181
     - :math:`\rm H_2 \ ^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_182
     - H2O_182
     - :math:`\rm HD^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_main_iso
     - H2O_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2O_pokazatel_main_iso
     - H2O_pokazatel_main_iso
     - Main isotopologue, Exomol, `Pokazatel et al. (2018) <https://doi.org/10.1093/mnras/sty1877>`_
     - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_
   * - H2S_main_iso
     - H2S_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2_12
     - H2_12
     - :math:`\rm HD`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - H2_main_iso
     - H2_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - HCN_main_iso
     - HCN_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - K
     - K
     - Main isotopologue, VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - NH3_main_iso
     - NH3_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - NH3_main_iso
     - NH3_main_iso
     - Main isotopologue, Exomol, `Yurchenko et al. (2011) <http://dx.doi.org/10.1111/j.1365-2966.2011.18261.x>`_
     - --
   * - NH3_Coles_main_iso
     - NH3_Coles_main_iso
     - Main isotopologue, Exomol, `Coles et al. (2019) <https://doi.org/10.1093/mnras/stz2778>`_
     - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_
   * - Na
     - Na
     - Main isotopologue, VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - O3_main_iso
     - O3_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - PH3_main_iso
     - PH3_main_iso
     - Main isotopologue, Exomol, `Sousa-Silva et al. (2014) <http://dx.doi.org/10.1093/mnras/stu2246>`_, converted from `DACE <https://dace.unige.ch/dashboard/>`_
     - `Adriano Miceli <adriano.miceli@stud.unifi.it>`_
   * - SiO_main_iso
     - SiO_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_all_iso
     - TiO_all_iso
     - All isotopologues, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_46_Plez
     - TiO_46_Plez
     - :math:`\rm \ ^{46}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_47_Plez
     - TiO_47_Plez
     - :math:`\rm \ ^{47}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_48_Plez
     - TiO_48_Plez
     - :math:`\rm \ ^{48}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_49_Plez
     - TiO_49_Plez
     - :math:`\rm \ ^{49}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_50_Plez
     - TiO_50_Plez
     - :math:`\rm \ ^{50}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - TiO_46_Exomol_McKemmish
     - TiO_46_Exomol_McKemmish
     - :math:`\rm \ ^{46}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - --
   * - TiO_47_Exomol_McKemmish
     - TiO_47_Exomol_McKemmish
     - :math:`\rm \ ^{47}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - --
   * - TiO_48_Exomol_McKemmish
     - TiO_48_Exomol_McKemmish
     - :math:`\rm \ ^{48}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - --
   * - TiO_49_Exomol_McKemmish
     - TiO_49_Exomol_McKemmish
     - :math:`\rm \ ^{49}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - --
   * - TiO_50_Exomol_McKemmish
     - TiO_50_Exomol_McKemmish
     - :math:`\rm \ ^{50}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - --
   * - VO
     - VO
     - Main isotopologue, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --
   * - VO_ExoMol_McKemmish
     - VO_ExoMol_McKemmish
     - `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
     - `S. de Regt <regt@strw.leidenuniv.nl>`_
   * - VO_ExoMol_Specific_Transitions
     - VO_ExoMol_Specific_Transitions
     - Most accurate transitions from `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
     - `S. de Regt <regt@strw.leidenuniv.nl>`_
   * - FeH_main_iso
     - FeH_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
     - --

*: see information box at the top of the page for mass fraction key handling.

Contributed atom and ion opacities, high resolution mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 10 10 10
   :header-rows: 1

   * - Name
     - Mass frac.*
     - Ref. line list / broad.
     - P (bar), T (K) range
     - Contributor
   * - Al
     - Al
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - B
     - B
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Be
     - Be
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Ca
     - Ca
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - CaII
     - CaII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Cr
     - Cr
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Fe
     - Fe
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - FeII
     - FeII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Li
     - Li
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Mg
     - Mg
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - MgII
     - MgII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - N
     - N
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Si
     - Si
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Ti
     - Ti
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - V
     - V
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - VII
     - VII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
   * - Y
     - Y
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_

*: see information box at the top of the page for mass fraction key handling.

Cloud opacities
_______________

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary
     - Description
   * - Al2O3(c)_cm
     - Al2O3(c)
     - Crystalline, Mie scattering (spherical)
   * - Al2O3(c)_cd
     - Al2O3(c)
     - Crystalline, DHS (irregular shape)
   * - Fe(c)_am
     - Fe(c)
     - Amorphous, Mie scattering (spherical)
   * - Fe(c)_ad
     - Fe(c)
     - Amorphous, DHS (irregular shape)
   * - Fe(c)_cm
     - Fe(c)
     - Crystalline, Mie scattering (spherical)
   * - Fe(c)_cd
     - Fe(c)
     - Crystalline, DHS (irregular shape)
   * - H2O(c)_cm
     - H2O(c)
     - Crystalline, Mie scattering (spherical)
   * - H2O(c)_cd
     - H2O(c)
     - Crystalline, DHS (irregular shape)
   * - KCL(c)_cm
     - KCL(c)
     - Crystalline, Mie scattering (spherical)
   * - KCL(c)_cd
     - KCL(c)
     - Crystalline, DHS (irregular shape)
   * - Mg05Fe05SiO3(c)_am
     - Mg05Fe05SiO3(c)
     - Amorphous, Mie scattering (spherical)
   * - Mg05Fe05SiO3(c)_ad
     - Mg05Fe05SiO3(c)
     - Amorphous, DHS (irregular shape)
   * - Mg2SiO4(c)_am
     - Mg2SiO4(c)
     - Amorphous, Mie scattering (spherical)
   * - Mg2SiO4(c)_ad
     - Mg2SiO4(c)
     - Amorphous, DHS (irregular shape)
   * - Mg2SiO4(c)_cm
     - Mg2SiO4(c)
     - Crystalline, Mie scattering (spherical)
   * - Mg2SiO4(c)_cd
     - Mg2SiO4(c)
     - Crystalline, DHS (irregular shape)
   * - MgAl2O4(c)_cm
     - MgAl2O4(c)
     - Crystalline, Mie scattering (spherical)
   * - MgAl2O4(c)_cd
     - MgAl2O4(c)
     - Crystalline, DHS (irregular shape)
   * - MgFeSiO4(c)_am
     - MgFeSiO4(c)
     - Amorphous, Mie scattering (spherical)
   * - MgFeSiO4(c)_ad
     - MgFeSiO4(c)
     - Amorphous, DHS (irregular shape)
   * - MgSiO3(c)_am
     - MgSiO3(c)
     - Amorphous, Mie scattering (spherical)
   * - MgSiO3(c)_ad
     - MgSiO3(c)
     - Amorphous, DHS (irregular shape)
   * - MgSiO3(c)_cm
     - MgSiO3(c)
     - Crystalline, Mie scattering (spherical)
   * - MgSiO3(c)_cd
     - MgSiO3(c)
     - Crystalline, DHS (irregular shape)
   * - Na2S(c)_cm
     - Na2S(c)
     - Crystalline, Mie scattering (spherical)
   * - Na2S(c)_cd
     - Na2S(c)
     - Crystalline, DHS (irregular shape)
   * - SiC(c)_cm
     - SiC(c)
     - Crystalline, Mie scattering (spherical)
   * - SiC(c)_cd
     - SiC(c)
     - Crystalline, DHS (irregular shape)


Rayleigh scatterers
___________________

.. list-table::
   :widths: 10 10
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary
   * - H2
     - H2
   * - He
     - He
   * - H2O
     - H2O
   * - CO2
     - CO2
   * - O2
     - O2
   * - N2
     - N2
   * - CO
     - CO
   * - CH4
     - CH4


Continuum opacity sources
_________________________

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary
     - Descripton
   * - H2-H2
     - H2
     - Collision induced absorption (CIA)
   * - H2-He
     - H2, He
     - Collision induced absorption (CIA)
   * - H2O-H2O
     - H2O
     - Collision induced absorption (CIA)
   * - H2O-N2
     - H2O, N2
     - Collision induced absorption (CIA)
   * - N2-H2
     - N2, H2
     - Collision induced absorption (CIA)
   * - N2-He
     - N2, He
     - Collision induced absorption (CIA)
   * - N2-N2
     - N2
     - Collision induced absorption (CIA)
   * - O2-O2
     - O2
     - Collision induced absorption (CIA)
   * - N2-O2
     - N2, O2
     - Collision induced absorption (CIA)
   * - CO2-CO2
     - CO2
     - Collision induced absorption (CIA)
   * - H-
     - H, H-, e-
     - H- bound-free and free-free opacity
