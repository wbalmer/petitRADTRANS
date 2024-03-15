==============================
WIP: Available opacity species
==============================
All the opacities that can be downloaded `via Keeper <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0>`_ in petitRADTRANS are listed below.

.. important:: Please cite the reference mentioned in the description (click the link) when making use of a line species listed below. Information about the opacity source are also available in the opacity HDF file under the key ``DOI`` and its attributes.

Line species
============
To add more line opacities, please see `Adding opacities <adding_opacities.html>`_, among them how to plug-and-play install the ExoMol opacities calculated in the pRT format, available from the `ExoMol website <http://www.ExoMol.com/>`_.

.. _lowResolution:

Low-resolution opacities (``"c-k"``, :math:`\lambda/\Delta\lambda=1000`)
------------------------------------------------------------------------
In correlated-k mode (``"c-k"``), most of the molecular opacities are calculated considering only the main isotopologue. Most of the time, the differences with including all isotopologues, at these resolving powers, are negligible (see comparison with `Baudino et al., 2017 <https://www.doi.org/10.3847/1538-4357/aa95be>`_).

For some species such as CO and TiO, the contribution of all isotopologues is considered, following their natural abundance **on Earth**. Some secondary isotopologues are also available. This has been done because of a large natural abundance ratio between the isotopes of some elements (e.g. Ti), and/or because of the significant spectral contribution of secondary isotopologues at the considered resolution (e.g. 12CO/13CO).

All ``c-k`` opacities referenced here have a resolving power of 1000 and cover **at least** wavelengths 0.3 to 50 µm. Pressure and temperature grids may vary. All of the opacities are sampled over 16 k-coefficients following the method described in `Baudino et al. (2015) <https://doi.org/10.1051/0004-6361/201526332>`_

.. important:: Correlated-k tables with the extension ``.ktable.petitRADTRANS.h5`` from `ExoMol <https://www.ExoMol.com/>`_ can be used directly.

The available correlated-k opacities are listed below. When multiple source are available for a species, the recommended one is indicated in bold.

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
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+18 <https://doi.org/10.1093/mnras/sty1524>`_
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
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - CaH
      - 40Ca-1H
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Li+12 <http://dx.doi.org/10.1016/j.jqsrt.2011.09.010>`_
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
      - HITEMP/Kurucz, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - **CO**
      - C-O-NatAbund__Chubb
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
      - `K. Chubb <klc20@st-andrews.ac.uk>`_
    * - 12CO
      - 12C-16O__HITEMP
      - HITEMP, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 13CO
      - 13C-16__HITEMP
      - HITEMP, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - **13CO**
      - 13C-16O__Li2015
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
      - `K. Chubb <klc20@st-andrews.ac.uk>`_
    * - CO2
      - 12C-16O2__UCL
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_
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
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Wende+10 <http://dx.doi.org/10.1051/0004-6361/201015220>`_
      - `K. Chubb <klc20@st-andrews.ac.uk>`_
    * - H2
      - 1H2__HITRAN
      - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
      - --
    * - H2O
      - 1H2-16O__HITEMP
      - HITEMP, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
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
      - VALD, Allard wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - K
      - 39K__Burrows
      - VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
      - --
    * - K
      - 39K_LorCut
      - VALD, Lorentzian wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
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
      - VALD, `Allard wings <https://ui.adsabs.harvard.edu/abs/2019yCat..36280120A/abstract>`_, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Na
      - 23Na__Burrows
      - VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
      - --
    * - Na
      - 23Na_LorCut
      - VALD, Lorentzian wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
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
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
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
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 48TiO
      - 48Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
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
      - B. Plez, see `Mollière+2019  <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - **VO**
      - 51V-16O__VOMYT
      - `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+16 <http://dx.doi.org/10.1093/mnras/stw1969>`_
      - `K. Chubb <klc20@st-andrews.ac.uk>`_

\*: discarding the spectral information.

.. _highResolution:

High resolution opacities (``"lbl"``, :math:`\lambda/\Delta\lambda=10^6`)
-------------------------------------------------------------------------
All ``lbl`` opacities referenced here have a resolving power of 1e6 and cover **at least** wavelengths 0.3 to 28 µm. Pressure and temperature grids may vary.

.. important:: Cross-section tables with the extension ``.xsec.TauREx.h5`` from `ExoMol <https://www.ExoMol.com/>`_ can be used directly.

The available line-by-line opacities are listed below. When multiple source are available for a species, the recommended one is indicated in bold.

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - Species name
      - Short file name*
      - Reference
      - Contributor
    * - Al **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - B **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Be **!!None!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - C2H2
      - 12C2-1H2__HITRAN
      - HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Ca **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - CaII **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Ca+
      - 40Ca_p__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - CaH
      - 40Ca-1H__MoLLIST
      - **???**
      - **???**
    * - CH3D **!!**
      - **!!None!!**
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - CH4
      - 12C-1H4__Hargreaves
      - HITEMP, `Hargreaves et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJS..247...55H/abstract>`_
      - --
    * - 13CH4
      - 13C-1H4__HITRAN
      - `HITRAN2019 <https://doi.org/10.1051/0004-6361/201935470>`_
      - **???**
    * - CO-NatAbund
      - C-O-NatAbund__HITRAN
      - see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - CO
      - 12C-16O__HITRAN
      - HITEMP, see `Mollière+2019  <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 12C-17O
      - 12C-17O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 12C-18O
      - 12C-18O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 13CO
      - 13C-16O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 13C-17O
      - 13C-17O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 13C-18O
      - 13C-18O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - CO2
      - 12-C-16O2__HITEMP
      - HITEMP, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Cr **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Fe **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - FeII **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - FeH
      - 56Fe-1H__MoLLIST
      - ExoMol, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - H2 **!!**
      - **!!None!!**
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - HD **!!**
      - **!!None!!**
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - H2O
      - 1H2-16O__HITEMP
      - HITEMP, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - **H2O**
      - 1H2-16O__POKAZATEL
      - ExoMol, `Pokazatel et al. (2018) <https://doi.org/10.1093/mnras/sty1877>`_
      - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_
    * - HDO
      - 1H-2H-16O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - H2-17O
      - 1H2-17O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - HD-17O
      - 1H-2H-17O
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - H2-18O
      - 1H2-18O__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - HD-18O
      - 1H-2H-18O
      - see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - H2S
      - 1H2-32S__HITRAN
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - HCN
      - 1H-12C-14N__Harris
      - Main isotopologue, ExoMol, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - **K**
      - 39K__Allard
      - VALD, Allard wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - K
      - 39K__Burrows
      - VALD,  `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
      - --
    * - K
      - 39K_LorCut
      - VALD, Lorentzian wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Li **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Mg **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - MgII **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - N **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - **Na**
      - 23Na__Allard
      - VALD, Allard wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Na
      - 23Na__Burrows
      - VALD,  `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
      - --
    * - Na
      - 23Na_LorCut
      - VALD, Lorentzian wings, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - NH3
      - 14N-1H3__HITRAN
      - ExoMol, `Yurchenko et al. (2011) <http://dx.doi.org/10.1111/j.1365-2966.2011.18261.x>`_
      - --
    * - **NH3**
      - 14N-1H3__CoYuTe
      - ExoMol, `Coles et al. (2019) <https://doi.org/10.1093/mnras/stz2778>`_
      - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_ (400--1600 K)
    * - O3 **!!**
      - **!!None!!**
      - HITRAN, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - OH
      - 16O-1H__MoLLIST
      - ExoMol, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - PH3
      - 31P-1H3__HITRAN
      - `HITRAN <https://doi.org/10.1016/j.jqsrt.2013.07.002>`_
      - --
    * - **PH3**
      - 31P-1H3__SAlTY
      - ExoMol, `Sousa-Silva et al. (2014) <http://dx.doi.org/10.1093/mnras/stu2246>`_, converted from `DACE <https://dace.unige.ch/dashboard/>`_
      - `Adriano Miceli <adriano.miceli@stud.unifi.it>`_
    * - Si **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - SiO
      - 28Si-16O__EBJT
      - ExoMol, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - Ti **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - TiO **???**
      - Ti-O-NatAbund__Toto
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO **???**
      - Ti-O-NatAbund__TotoMcKemmish
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_46_Plez **!!**
      - **!!None!!**
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_47_Plez **!!**
      - **!!None!!**
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_48_Plez **???**
      - **TiO_48???**
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_49_Plez **!!**
      - **!!None!!**
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_50_Plez **!!**
      - **!!None!!**
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO_46_Exomol_McKemmish **!!**
      - **!!None!!**
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - TiO_47_Exomol_McKemmish **???**
      - **TiO_47_exo_new???**
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - TiO_48_Exomol_McKemmish **???**
      - **TiO_48_exo_new???**
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - TiO_49_Exomol_McKemmish **!!**
      - **!!None!!**
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - TiO_50_Exomol_McKemmish **!!**
      - **!!None!!**
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - V **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - VII **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - VO
      - 51V-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - VO_ExoMol_McKemmish **!!**
      - **!!None!!**
      - `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
      - `S. de Regt <regt@strw.leidenuniv.nl>`_
    * - VO_ExoMol_Specific_Transitions **!!**
      - **!!None!!**
      - Most accurate transitions from `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
      - `S. de Regt <regt@strw.leidenuniv.nl>`_
    * - Y **!!**
      - **!!None!!**
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_

\*: discarding the spectral information.

.. _namingConvention:

File naming convention
----------------------
In petitRADTRANS, line species opacities follow a naming convention identical to that of `ExoMol <https://www.ExoMol.com/>`_. The isotopes are explicitly displayed, for example, ``13C-16O`` means a CO molecule with a carbon-13 and an oxygen-16 atom. When the opacity corresponds to a mixture of isotopologues, the flag ``NatAbund`` is used.

Note that writing the full file opacity name when using a ``Radtrans``-like object is not necessary, as partial naming is allowed. When no isotopic information is given, the main isotopologue is picked (e.g. ``H2O`` is equivalent to ``1H2-16O``).

.. important:: The ``line_species`` opacity name and the ``mass_fractions`` dictionary keys must match *exactly*.

Below are some working opacity name examples:

- File names:

    * ``1H2-16O__POKAZATEL.R1000_0.1-250mu.ktable.petitRADTRANS.h5``
    * ``C-O-NatAbund__HITEMP.R250_0.1-250mu.ktable.petitRADTRANS.h5``
    * ``1H-12C-14N__Harris.R1e6_0.3-28mu.xsec.petitRADTRANS.h5``
    * ``39K__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5``

- Names valid in scripts:

    * ``H2O``
    * ``H2O__POKAZATEL``
    * ``H2O.R1000``
    * ``H2-17O``
    * ``CO-NatAbund``
    * ``Ca+``
    * ``1H-2H-18O__HITEMP.R1e6_0.3-28mu``

Hereafter are the explicit file naming rules for line species:

- Species names are based on their chemical formula.
- Elements in the chemical formula are separated by ``-``.
- The number in front of the element indicates its isotope, when relevant.
- The number after the element indicates its quantity in the molecule, when relevant.
- Opacities combining isotopologues following their natural (i.e. Earth) abundance are indicated with the string ``-NatAbund`` after the chemical formula. In that case, no isotope number should be present next to the elements.
- The charge of the species is indicated after the formula, starting with ``_``. The character ``p`` is used for positive charges and ``n`` for negative charges.
- The number in front of the charge indicates the charge amount.
- The source of the opacity is indicated after the charge, starting with ``__``.
- The spectral information of the opacity is indicated after the source, starting with ``.``.
- The character ``R`` indicates constant resolving power (:math:`\lambda/\Delta\lambda` constant).
- The string ``DeltaWavenumber`` indicates constant spacing in wavenumber (:math:`\Delta\nu` constant).
- The string ``DeltaWavelength`` indicates constant spacing in wavelength (:math:`\Delta\lambda` constant).
- The number coming after the above indicates the spacing.
- The wavelength range, in µm, is indicated afterward, starting with a ``_`` and ending with ``mu``. The upper and lower boundaries are separated with ``-``.
- The nature of the opacity is indicated afterward, starting with a ``.``. It is ``ktable`` for correlated-k opacities, and ``xsec`` for line-by-line opacities.
- The extension of the file is always ``.petitRADTRANS.h5``.

.. _continuum:

Gas continuum opacity sources
=============================

Available collision-induced absorptions
---------------------------------------
The available collision-induced absorptions are listed below.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Species name
      - File name
      - Reference
    * - CO2--CO2
      - **???**
      - **???**
    * - H2--H2
      - H2--H2-NatAbund__BoRi.R831_0.6-250mu
      - `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
    * - H2--He
      - H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu
      - `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
    * - H2O--H2O
      - **???**
      - **???**
    * - H2O--N2
      - **???**
      - **???**
    * - N2--H2
      - **???**
      - **???**
    * - N2--He
      - **???**
      - **???**
    * - N2--N2
      - **???**
      - **???**
    * - N2--O2
      - **???**
      - **???**
    * - O2--O2
      - **???**
      - **???**

Other gas continuum contributors
--------------------------------
In addition to CIA, petitRADTRANS can also calculate the H- (bound-free and free-free) absorptions. In that case, the ``H-`` string must be present in the ``gas_continuum_contributors`` list. In the ``mass_fractions`` dictionary, the keys ``H-`` and ``e-`` must be present as well.

File naming convention
----------------------
Gas continuum sources follow a naming convention similar to that of the :ref:`line species<namingConvention>`. For collision-induced absorptions (CIA), the 2 colliding species are separated with ``--``.

Most of the CIA are given for species with their Earth natural isotopologue abundances. The very low resolving power of those opacities makes isotope-specific data irrelevant.

.. important:: If a ``gas_continuum_contributors`` opacity name refer to a single species, it must be added to the ``mass_fractions`` dictionary. If a ``gas_continuum_contributors`` opacity name is a CIA, the ``mass_fractions`` dictionary keys must contains the colliding species.

Below are some working opacity name examples:

- File names:

    * ``H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5``
    * ``H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu``

- Names valid in scripts:

    * ``H2-H2``
    * ``H2--He``
    * ``He-H2``
    * ``H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu``

Hereafter are the explicit file naming rules for line species:

- Gas continuum species names follow the same convention as the :ref:`line species<namingConvention>`, with the following additions.
- For collision induced absorptions, the two colliding species are separated with ``--``. The ``-NatAbund`` flag must be placed after the two species.
- The extension of the file is always ``.ciatable.petitRADTRANS.h5``.

.. _clouds:

Cloud opacities
===============

Available cloud opacities
-------------------------
All clouds opacities referenced here have a resolving power of 39 and cover **at least** wavelengths 0.1 to 250 µm. Particle size grid may vary.

All solid condensate opacities listed are available for both the DHS and Mie scattering particle shapes.

.. important:: Currently no space group information are given for the crystal species. **We plan to add them in the future.**

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Species name to be handed to pRT object
      - Long file name
      - Reference for optical data (mostly DOIs)
    * - Al2O3(s)_crystalline__DHS
      - Al2-O3-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1006/icar.1995.1055
    * - Al2O3(s)_crystalline__Mie
      - Al2-O3-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1006/icar.1995.1055
    * - C(s)_crystalline__DHS
      - C-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Draine (2003), AJ., 598:1026
    * - C(s)_crystalline__Mie
      - C-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Draine (2003), AJ., 598:1026
    * - CaTiO3(s)_crystalline__DHS
      - Ca-Ti-O3-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Posch et al. (2003), Ap&SS, 149:437; Ueda et al 1998 J. Phys.: Condens. Matter 10 3669; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - CaTiO3(s)_crystalline__Mie
      - Ca-Ti-O3-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Posch et al. (2003), Ap&SS, 149:437; Ueda et al 1998 J. Phys.: Condens. Matter 10 3669; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Cr(s)_structureUnclear__DHS
      - Cr-NatAbund(s)_structureUnclear__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Lynch&Hunter (1991) in Palik: "Handbook of Optical Constants of Solids"; Rakic et al. (1998) Applied Optics Vol. 37, Issue 22
    * - Cr(s)_structureUnclear__Mie
      - Cr-NatAbund(s)_structureUnclear__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Lynch&Hunter (1991) in Palik: "Handbook of Optical Constants of Solids"; Rakic et al. (1998) Applied Optics Vol. 37, Issue 22
    * - Fe(s)_amorphous__DHS
      - Fe-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1086/173677
    * - Fe(s)_amorphous__Mie
      - Fe-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1086/173677
    * - Fe(s)_crystalline__DHS
      - Fe-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1996A&A...311..291H
    * - Fe(s)_crystalline__Mie
      - Fe-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1996A&A...311..291H
    * - Fe2O3(s)_structureUnclear__DHS
      - Fe2-O3-NatAbund(s)_structureUnclear__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Amaury H.M.J. Triaud, in Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2O3(s)_structureUnclear__Mie
      - Fe2-O3-NatAbund(s)_structureUnclear__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Amaury H.M.J. Triaud, in Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2SiO4(s)_structureUnclear__DHS
      - Fe2-Si-O4-NatAbund(s)_structureUnclear__DHS.R39_0.4-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Fabian et al. (2001), A&A Vol. 378; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2SiO4(s)_structureUnclear__Mie
      - Fe2-Si-O4-NatAbund(s)_structureUnclear__Mie.R39_0.4-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Fabian et al. (2001), A&A Vol. 378; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - FeO(s)_crystalline__DHS
      - Fe-O-NatAbund(s)_crystalline_000__DHS.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Henning et al. (1995), Astronomy and Astrophysics Supplement, v.112, p.143; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - FeO(s)_crystalline__Mie
      - Fe-O-NatAbund(s)_crystalline_000__Mie.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Henning et al. (1995), Astronomy and Astrophysics Supplement, v.112, p.143; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - FeS(s)_crystalline__DHS
      - Fe-S-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Pollack et al. (1994) ApJ, 421:615; Henning&Mutschke (1997), A&A, 327:743
    * - FeS(s)_crystalline__Mie
      - Fe-S-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Pollack et al. (1994) ApJ, 421:615; Henning&Mutschke (1997), A&A, 327:743
    * - H2O(l)__Mie
      - H2-O-NatAbund(l)__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - URI http://hdl.handle.net/10355/11599 : Segelstein, D. J. 1981, Master Thesis, University of Missouri-Kansas City, USA
    * - H2O(s)_crystalline__DHS
      - H2-O-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1093/mnras/271.2.481
    * - H2O(s)_crystalline__Mie
      - H2-O-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1093/mnras/271.2.481
    * - H2SO4(l)__Mie-25-weight-percent-aqueous
      - H2-S-O4-NatAbund(l)__Mie-25-weight-percent-aqueous.R39_2.5-25mu.cotable.petitRADTRANS.h5
      - 10.1364/AO.14.000208
    * - H2SO4(l)__Mie-50-weight-percent-aqueous
      - H2-S-O4-NatAbund(l)__Mie-50-weight-percent-aqueous.R39_2.5-25mu.cotable.petitRADTRANS.h5
      - 10.1364/AO.14.000208
    * - H2SO4(l)__Mie-75-weight-percent-aqueous
      - H2-S-O4-NatAbund(l)__Mie-75-weight-percent-aqueous.R39_2.5-25mu.cotable.petitRADTRANS.h5
      - 10.1364/AO.14.000208
    * - H2SO4(l)__Mie-85-weight-percent-aqueous
      - H2-S-O4-NatAbund(l)__Mie-85-weight-percent-aqueous.R39_2.5-25mu.cotable.petitRADTRANS.h5
      - 10.1364/AO.14.000208
    * - H2SO4(l)__Mie-96-weight-percent-aqueous
      - H2-S-O4-NatAbund(l)__Mie-96-weight-percent-aqueous.R39_2.5-25mu.cotable.petitRADTRANS.h5
      - 10.1364/AO.14.000208
    * - KCl(s)_crystalline__DHS
      - K-Cl-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Edward D. Palik: Handbook of Optical Constants of Solids, Elsevier Science, 2012
    * - KCl(s)_crystalline__Mie
      - K-Cl-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Edward D. Palik: Handbook of Optical Constants of Solids, Elsevier Science, 2012
    * - Mg05Fe05SiO3(s)_amorphous__DHS
      - Mg05-Fe05-Si-O3-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J
    * - Mg05Fe05SiO3(s)_amorphous__Mie
      - Mg05-Fe05-Si-O3-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J
    * - Mg2SiO4(s)_amorphous__DHS
      - Mg2-Si-O4-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - Mg2SiO4(s)_amorphous__Mie
      - Mg2-Si-O4-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - Mg2SiO4(s)_crystalline__DHS
      - Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1002/pssb.2220550224
    * - Mg2SiO4(s)_crystalline__Mie
      - Mg2-Si-O4-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1002/pssb.2220550224
    * - MgAl2O4(s)_crystalline__DHS
      - Mg-Al2-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Edward D. Palik: Handbook of Optical Constants of Solids, Elsevier Science, 2012
    * - MgAl2O4(s)_crystalline__Mie
      - Mg-Al2-O4-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Edward D. Palik: Handbook of Optical Constants of Solids, Elsevier Science, 2012
    * - MgFeSiO4(s)_amorphous__DHS
      - Mg-Fe-Si-O4-NatAbund(s)_amorphous__DHS.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Dorschner et al. (1995), A&A Vol. 300; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - MgFeSiO4(s)_amorphous__Mie
      - Mg-Fe-Si-O4-NatAbund(s)_amorphous__Mie.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Dorschner et al. (1995), A&A Vol. 300; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - MgO(s)_crystalline__DHS
      - Mg-O-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Roessler & Huffman (1981) in Palik: "Handbook of Optical Constants of Solids"
    * - MgO(s)_crystalline__Mie
      - Mg-O-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Roessler & Huffman (1981) in Palik: "Handbook of Optical Constants of Solids"
    * - MgSiO3(s)_amorphous__DHS
      - Mg-Si-O3-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - MgSiO3(s)_amorphous__Mie
      - Mg-Si-O3-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - MgSiO3(s)_crystalline__DHS
      - Mg-Si-O3-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1998A&A...339..904J, 10.1086/192321
    * - MgSiO3(s)_crystalline__Mie
      - Mg-Si-O3-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1998A&A...339..904J, 10.1086/192321
    * - MnS(s)_structureUnclear__DHS
      - Mn-S-NatAbund(s)_structureUnclear__DHS.R39_0.1-190mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Huffman&Wild (1967) Phys. Rev., Vol 156:989; Montaner et al. (1979) Phys. Status Solidi Appl. Res., Vol. 52:597
    * - MnS(s)_structureUnclear__Mie
      - Mn-S-NatAbund(s)_structureUnclear__Mie.R39_0.1-190mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Huffman&Wild (1967) Phys. Rev., Vol 156:989; Montaner et al. (1979) Phys. Status Solidi Appl. Res., Vol. 52:597
    * - Na2S(s)_crystalline__DHS
      - Na2-S-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1088/0004-637X/756/2/172
    * - Na2S(s)_crystalline__Mie
      - Na2-S-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1088/0004-637X/756/2/172
    * - NaCl(s)_crystalline__DHS
      - Na-Cl-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Eldrige & Palik (1985) in Palik: "Handbook of Optical Constants of Solids"
    * - NaCl(s)_crystalline__Mie
      - Na-Cl-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Eldrige & Palik (1985) in Palik: "Handbook of Optical Constants of Solids"
    * - SiC(s)_crystalline__DHS
      - Si-C-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1988A&A...194..335P
    * - SiC(s)_crystalline__Mie
      - Si-C-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1988A&A...194..335P
    * - SiO(s)_amorphous__DHS
      - Si-O-NatAbund(s)_amorphous__DHS.R39_0.1-100mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Wetzel et al. (2013) A&A, Vol 553:A92
    * - SiO(s)_amorphous__Mie
      - Si-O-NatAbund(s)_amorphous__Mie.R39_0.1-100mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Wetzel et al. (2013) A&A, Vol 553:A92
    * - SiO2(s)_amorphous__DHS
      - Si-O2-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Henning&Mutschke (1997), A&A Vol. 327; Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - SiO2(s)_amorphous__Mie
      - Si-O2-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Henning&Mutschke (1997), A&A Vol. 327; Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - SiO2(s)_crystalline__DHS
      - Si-O2-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Zeidler et al. (2013), A&A, Vol. 553:A81; Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - SiO2(s)_crystalline__Mie
      - Si-O2-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Zeidler et al. (2013), A&A, Vol. 553:A81; Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - TiC(s)_crystalline__DHS
      - Ti-C-NatAbund(s)_crystalline_000__DHS.R39_0.1-207mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Koide et al 1990, Phys Rev B, 42,4979; Henning & Mutschke 2001, Spec. Acta Part A57, 815
    * - TiC(s)_crystalline__Mie
      - Ti-C-NatAbund(s)_crystalline_000__Mie.R39_0.1-207mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Koide et al 1990, Phys Rev B, 42,4979; Henning & Mutschke 2001, Spec. Acta Part A57, 815
    * - TiO2(s)_crystalline__DHS
      - Ti-O2-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Zeidler et al. (2011), A&A 526:A68; Posch et al. (2003), Ap&SS, 149:437; Siefke et al. (2016),  Adv. Opt. Mater. 4:1780; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - TiO2(s)_crystalline__Mie
      - Ti-O2-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Zeidler et al. (2011), A&A 526:A68; Posch et al. (2003), Ap&SS, 149:437; Siefke et al. (2016),  Adv. Opt. Mater. 4:1780; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - ZnS(s)_crystalline__DHS
      - Zn-S-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Palik & Addamiano (1985) in Palik: "Handbook of Optical Constants of Solids"
    * - ZnS(s)_crystalline__Mie
      - Zn-S-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Palik & Addamiano (1985) in Palik: "Handbook of Optical Constants of Solids"

File naming convention
----------------------
Cloud species follow a naming convention similar to that of the :ref:`line species<namingConvention>`. In addition to the species name, the state of matter and other condensate-specific information are added. Partial naming is  also allowed when using ``Radtrans``-like objects.

Most of the condensate species opacities are given for their Earth natural isotopologue abundances. The very low resolving power of those opacities makes isotope-specific data irrelevant.

The source indication (after ``__`` in the file name) is used to indicate the method of the opacity calculation:
- ``DHS`` stands for "Double-shelled Hollow Spheres" particles. Opacities calculated with this particle shape are generally considered more realistic.
- ``Mie`` stands for spherical particles, (opacities calculated with Mie Scattering).

.. important::
     The ``cloud_species`` opacity name and the ``mass_fractions`` dictionary keys must match *exactly*.

Below are some working opacity name examples:

* File names:

  * ``Mg2-Si-O4-NatAbund(s)_crystalline_062__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5``
  * ``H2-O-NatAbund(l)__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5``
  * ``Fe-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5``

* Names valid in scripts:

  * ``Mg2SiO4(s)_crystalline``
  * ``Mg2SiO4(s)_amorphous``
  * ``H2O(l)``
  * ``Fe(s)_crystalline__DHS``
  * ``H2-O-NatAbund(s)_crystalline_194__Mie.R39_0.1-250mu``

Hereafter are the explicit file naming rules for line species:

- Cloud species names follow the same convention as the :ref:`line species<namingConvention>`, with the following additions.
- After the full chemical formula and the ``-NatAbund`` flag, if relevant, the physical state of the condensate is indicated between parenthesis: ``(s)`` for solids, ``(l)`` for liquids
- For **solid** condensates **only**, after the state:

    * the internal structure of the condensate particles is indicated after a ``_``, it can be either ``crystalline`` or ``amorphous``,
    * in the rare case where the internal structure of the condensate particles is not indicated by the source providing the opacities, the label ``unclearStructure`` is used instead,
    * for ``amorphous`` solids, a string indicating the amorphous state in front of a ``_`` **can** be added,
    * for ``crystalline`` solids, 3 numbers in front of a ``_`` **must** be added, indicating the `space group <https://en.wikipedia.org/wiki/List_of_space_groups>`_,
    * when the space group of crystals is not provided by the source or has not been verified yet, the number ``000`` is used (space group number range from ``001`` to ``230``).

- For **liquid** condensates, the above requirements for solids do not apply.
- The source and spectral information that follows obey the same rules as for the line species.
- The extension of the file is always ``.cotable.petitRADTRANS.h5``.

Rayleigh scatterers
===================
In contrast with the above opacities, Rayleigh scattering cross-sections are are not stored into files. Instead, the cross-sections are calculated using wavelength-dependent best-fit parameters to measurements (see sources below) on-the-fly in petitRADTRANS.

.. caution::
    For the high resolution mode of pRT (``mode='lbl'``) the numerical cost of calculating Rayleigh cross sections becomes noticeable. Currently, the H2 and He Rayleigh scattering cross-sections benefit from an optimised code and are faster to calculate than the other listed species.

    **We intend to optimise all the Rayleigh scattering absorption calculations in a future update**.

    For low-resolution calculations (``mode='c-k'``) the cost of calculating Rayleigh cross sections is negligible.

The Rayleigh scattering cross-sections available in pRT are listed below:

- CH4 (`Sneep & Ubachs 2005 <https://ui.adsabs.harvard.edu/abs/2005JQSRT..92..293S/abstract>`_)
- CO (`Sneep & Ubachs 2005 <https://ui.adsabs.harvard.edu/abs/2005JQSRT..92..293S/abstract>`_)
- CO2 (`Sneep & Ubachs 2005 <https://ui.adsabs.harvard.edu/abs/2005JQSRT..92..293S/abstract>`_)
- **H2** (`Dalgarno & Williams 1962 <https://ui.adsabs.harvard.edu/abs/1962ApJ...136..690D/abstract>`_)
- H2O (`Harvey et al. 1998 <https://ui.adsabs.harvard.edu/abs/1998JPCRD..27..761H/abstract>`_)
- **He** (`Chan & Dalgarno 1965 <https://ui.adsabs.harvard.edu/abs/1965PPS....85..227C/abstract>`_)
- N2 (`Thalmann et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014JQSRT.147..171T/abstract>`_, `2017 <https://ui.adsabs.harvard.edu/abs/2017JQSRT.189..281T/abstract>`_)
- O2 (`Thalmann et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014JQSRT.147..171T/abstract>`_, `2017 <https://ui.adsabs.harvard.edu/abs/2017JQSRT.189..281T/abstract>`_)
