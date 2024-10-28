=========================
Available opacity species
=========================
All the opacities that can be downloaded `via Keeper <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0>`_ in petitRADTRANS are listed below. petitRADTRANS will automatically download opacity tables if you request a species that is not on your hard drive yet, but available on Keeper.

.. Tip:: Additional sources of opacities, and how to calculate and add your own, are described in `"Adding opacities" <adding_opacities.html>`_.

.. important:: **Converting pRT2 opacities:** if you added opacities to pRT yourself in the past, before pRT3 was released (May 2024): these need to be converted to pRT3 format. This is explained in `"Converting pRT2 opacities to pRT3 format" <pRT3_changes_description.html#converting-prt2-opacities-to-prt3-format>`_.

.. important:: Please cite the reference mentioned in the description when making use of a line species listed below. Information about the opacity source are also available in the opacity HDF file under the key ``DOI`` and its attributes.

Line species
============
To add more line opacities in addition to what is listed below, please see `"Adding opacities" <adding_opacities.html>`_, among them how to plug-and-play install the ExoMol opacities calculated in the pRT format, available from the `ExoMol website <https://www.exomol.com/data/data-types/opacity/>`_.

.. _lowResolution:

Low-resolution opacities (``"c-k"``, :math:`\lambda/\Delta\lambda=1000`)
------------------------------------------------------------------------
In correlated-k mode (``"c-k"``), most of the molecular opacities are calculated considering only the main isotopologue. Most of the time, the differences with including all isotopologues, at these resolving powers, are negligible (see comparison with `Baudino et al., 2017 <https://www.doi.org/10.3847/1538-4357/aa95be>`_).

For some species such as CO and TiO, the contribution of all isotopologues is considered, following their natural abundance **on Earth**. Some secondary isotopologues are also available. This has been done because of a large natural abundance ratio between the isotopes of some elements (e.g. Ti), and/or because of the significant spectral contribution of secondary isotopologues at the considered resolution (e.g. 12CO/13CO).

All ``c-k`` opacities referenced here have a resolving power of 1000 and cover **at least** wavelengths 0.3 to 50 µm. The actual wavelength coverage is given by looking at the full filenames in the table below.
Pressure and temperature grids may vary and are thus treated on a per-species basis within pRT. All opacities are stored at 16 discrete g values of the cumulative opacity distribution function, per spectral bin, as described in `Mollière et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_, their Section 3.1.

.. Tip:: Correlated-k tables with the extension ``.ktable.petitRADTRANS.h5`` from `ExoMol <https://www.exomol.com/data/data-types/opacity/>`_ can be used directly.

The available correlated-k opacities are listed below. When multiple source are available for a species, the recommended one is indicated in bold.

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Short species name [#]_
      - Unique species name [#]_
      - File name
      - Reference for line list (mostly DOIs)
      - Contributor
    * - Al
      - 27Al__Kurucz
      - 27Al__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - AlH
      - 27Al-1H__AlHambra
      - 27Al-1H__AlHambra.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/sty1524
      - --
    * - AlO
      - 27Al-16O__ATP
      - 27Al-16O__ATP.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stv507
      - --
    * - Al+
      - 27Al_+__Kurucz
      - 27Al_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - C2H2
      - 12C2-1H2__aCeTY
      - 12C2-1H2__aCeTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/staa229
      - --
    * - C2H4
      - 12C2-1H4__MaYTY
      - 12C2-1H4__MaYTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/sty1239
      - --
    * - 12C-1H3-2H
      - 12C-1H3-2H__HITRAN
      - 12C-1H3-2H__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - CH4
      - 12C-1H4__HITEMP
      - 12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.3847/1538-4365/ab7a1a
      - --
    * - CH4
      - 12C-1H4__YT34to10
      - 12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201731026
      - --
    * - 13C-1H3-2H
      - 13C-1H3-2H__HITRAN
      - 13C-1H3-2H__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - 13C-1H4
      - 13C-1H4__HITRAN
      - 13C-1H4__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - CO
      - 12C-16O__HITEMP
      - 12C-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - 13C-16O
      - 13C-16O__HITEMP
      - 13C-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - 13C-16O
      - 13C-16O__Li2015
      - 13C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5
      - 10.1088/0067-0049/216/1/15
      - --
    * - C-O-NatAbund
      - C-O-NatAbund__Chubb
      - C-O-NatAbund__Chubb.R1000_0.3-50mu.ktable.petitRADTRANS.h5
      - 10.1088/0067-0049/216/1/15
      - --
    * - C-O-NatAbund
      - C-O-NatAbund__HITEMP
      - C-O-NatAbund__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - CO2
      - 12C-16O2__UCL-4000
      - 12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/staa1874
      - --
    * - C-S2-NatAbund
      - C-S2-NatAbund__HITRAN
      - C-S2-NatAbund__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - CaH
      - 40Ca-1H__MoLLIST
      - 40Ca-1H__MoLLIST.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2011.09.010
      - --
    * - Ca+
      - 40Ca_+__Kurucz
      - 40Ca_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - CrH
      - 52Cr-1H__MoLLIST
      - 52Cr-1H__MoLLIST.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1086/342242
      - --
    * - Fe
      - 56Fe__Kurucz
      - 56Fe__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - FeH
      - 56Fe-1H__MoLLIST
      - 56Fe-1H__MoLLIST.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201015220
      - --
    * - Fe+
      - 56Fe_+__Kurucz
      - 56Fe_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - H2
      - 1H2__HITRAN
      - 1H2__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 1H-2H-16O
      - 1H-2H-16O__HITRAN
      - 1H-2H-16O__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2O
      - 1H2-16O__HITEMP
      - 1H2-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - H2O
      - 1H2-16O__POKAZATEL
      - 1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5
      - 10.1093/mnras/sty1877
      - --
    * - 1H2-17O
      - 1H2-17O__HITRAN
      - 1H2-17O__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 1H2-18O
      - 1H2-18O__HITRAN
      - 1H2-18O__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2S
      - 1H2-32S__AYT2
      - 1H2-32S__AYT2.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stw1133
      - --
    * - HCN
      - 1H-12C-14N__Harris
      - 1H-12C-14N__Harris.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stt2011
      - --
    * - K
      - 39K__Allard
      - 39K__Allard.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201935470
      - --
    * - K
      - 39K__Burrows
      - 39K__Burrows.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1086/345412
      - --
    * - K
      - 39K__LorCut
      - 39K__LorCut.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://vald.astro.uu.se/
      - --
    * - Li
      - 7Li__Kurucz
      - 7Li__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - Mg
      - 24Mg__Kurucz
      - 24Mg__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - MgH
      - 24Mg-1H__MoLLIST
      - 24Mg-1H__MoLLIST.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stt510
      - --
    * - MgO
      - 24Mg-16O__LiTY
      - 24Mg-16O__LiTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stz912
      - --
    * - Mg+
      - 24Mg_+__Kurucz
      - 24Mg_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - NH3
      - 14N-1H3__CoYuTe
      - 14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stz2778
      - --
    * - 15N-1H3
      - 15N-1H3__HITRAN
      - 15N-1H3__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - Na
      - 23Na__NewAllard
      - 23Na__NewAllard.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201935593
      - `S. der Regt <mailto:regt@strw.leidenuniv.nl>`_
    * - Na
      - 23Na__Burrows
      - 23Na__Burrows.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1086/345412
      - --
    * - Na
      - 23Na__LorCut
      - 23Na__LorCut.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://vald.astro.uu.se/
      - --
    * - NaH
      - 23Na-1H__Rivlin
      - 23Na-1H__Rivlin.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stv979
      - --
    * - O
      - 16O__Kurucz
      - 16O__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - 16O-17O
      - 16O-17O__HITRAN
      - 16O-17O__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - 16O-18O
      - 16O-18O__HITRAN
      - 16O-18O__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - O2
      - 16O2__HITRAN
      - 16O2__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - O3
      - 16O3__HITRAN
      - 16O3__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - OH
      - 16O-1H__HITEMP
      - 16O-1H__HITEMP.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1016/j.jqsrt.2018.06.016
      - --
    * - PH3
      - 31P-1H3__SAlTY
      - 31P-1H3__SAlTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stu2246
      - --
    * - SH
      - 32S-1H__GYT
      - 32S-1H__GYT.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/sty939
      - --
    * - SO2
      - 32S-16O2__ExoAmes
      - 32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stw849
      - --
    * - Si
      - 28Si__Kurucz
      - 28Si__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - SiO
      - 28Si-16O__SiOUVenIR
      - 28Si-16O__SiOUVenIR.R1000_0.1-50mu.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stab3267
      - --
    * - SiO2
      - 28Si-16O2__OYT3
      - 28Si-16O2__OYT3.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - www.exomol.com/data/molecules/SiO2/28Si-16O2/OYT3
      - --
    * - Si+
      - 28Si_+__Kurucz
      - 28Si_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - Ti
      - 48Ti__Kurucz
      - 48Ti__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - TiO
      - 48Ti-16O__McKemmish
      - 48Ti-16O__McKemmish.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stz1818
      - `Chubb et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `ExoMolOP <https://www.exomol.com/data/data-types/opacity/>`_
    * - TiO
      - 48Ti-16O__Plez
      - 48Ti-16O__Plez.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201935470
      - --
    * - Ti-O-NatAbund
      - Ti-O-NatAbund__McKemmish
      - Ti-O-NatAbund__McKemmish.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stz1818
      - `Chubb et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `ExoMolOP <https://www.exomol.com/data/data-types/opacity/>`_
    * - Ti-O-NatAbund
      - Ti-O-NatAbund__Plez
      - Ti-O-NatAbund__Plez.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201935470
      - --
    * - Ti+
      - 48Ti_+__Kurucz
      - 48Ti_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - V
      - 51V__Kurucz
      - 51V__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_
    * - VO
      - 51V-16O__Plez
      - 51V-16O__Plez.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - 10.1051/0004-6361/201935470
      - --
    * - VO
      - 51V-16O__VOMYT
      - 51V-16O__VOMYT.R1000_0.3-50mu.ktable.petitRADTRANS.h5.ktable.petitRADTRANS.h5
      - 10.1093/mnras/stw1969
      - --
    * - V+
      - 51V_+__Kurucz
      - 51V_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5
      - http://kurucz.harvard.edu/
      - `K. Molaverdikhani <mailto:karan.molaverdikhani@colorado.edu>`_

.. [#] This is the "minimal name" you have to provide pRT with in order to be able to load this opacity. If there are multiple options (e.g., you request ``'CO'``, but there is the HITEMP and the Exomol line list), it will ask you which one you prefer.

.. [#] This is the unique name for which there is no source ambiguity, when requested in pRT. The default resolving power and wavelength range will be used, unless more information are given.

.. _highResolution:

High resolution opacities (``"lbl"``, :math:`\lambda/\Delta\lambda=10^6`)
-------------------------------------------------------------------------
All ``lbl`` opacities referenced here have a wavelength binning of :math:`\lambda/\Delta\lambda=10^6` and all files cover wavelengths from 0.3 to 28 µm **exactly**. We are currently working on a version that allows variable wavelength ranges per species, as already implemented for the ``c-k`` mode. Pressure and temperature grids may vary.

.. important:: TauREx' cross-section tables with the extension ``.xsec.TauREx.h5`` from `ExoMol <https://www.exomol.com/data/data-types/opacity/>`_ can be used directly, but these have a lower wavelength binning :math:`\lambda/\Delta\lambda=15,000`, so should only be used for data with a spectral resolution :math:`R\lesssim 150`, to avoid opacity sampling noise.

The available line-by-line opacities are listed below. When multiple source are available for a species, the recommended one is indicated in bold.

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - Short species name [#]_
      - Unique species name [#]_
      - Reference
      - Contributor
    * - Al
      - 27Al__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - B
      - 11B__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Be
      - 9Be__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - C2H2
      - 12C2-1H2__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - Ca
      - 40Ca__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Ca+
      - 40Ca_p__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - CaH
      - 40Ca-1H__MoLLIST
      - `Li et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012JQSRT.113...67L/abstract>`_
      - --
    * - CH3D
      - 12C-1H3-2H__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - CH4
      - 12C-1H4__Hargreaves
      - HITEMP, `Hargreaves et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJS..247...55H/abstract>`_
      - --
    * - 13CH4
      - 13C-1H4__HITRAN
      - 10.1016/j.jqsrt.2021.107949
      - --
    * - CO-NatAbund
      - C-O-NatAbund__HITEMP
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - CO
      - 12C-16O__HITEMP
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - 12C-17O
      - 12C-17O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 12C-18O
      - 12C-18O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 13CO
      - 13C-16O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 13C-17O
      - 13C-17O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - 13C-18O
      - 13C-18O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - CO2
      - 12-C-16O2__HITEMP
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - Cr
      - 52Cr__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Fe
      - 56Fe__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Fe+
      - 56Fe_p__Kurucz.
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - FeH
      - 56Fe-1H__MoLLIST
      - 10.1016/j.jqsrt.2019.106687
      - --
    * - H2
      - 1H2__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - HD
      - 1H-2H__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2O
      - 1H2-16O__HITEMP
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - H2O
      - 1H2-16O__POKAZATEL
      - ExoMol, `Pokazatel et al. (2018) <https://doi.org/10.1093/mnras/sty1877>`_
      - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_
    * - HDO
      - 1H-2H-16O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2-17O
      - 1H2-17O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - HD-17O
      - 1H-2H-17O
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2-18O
      - 1H2-18O__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - HD-18O
      - 1H-2H-18O
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - H2S
      - 1H2-32S__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - HCN
      - 1H-12C-14N__Harris
      - 10.1111/j.1365-2966.2005.09960.x
      - --
    * - K
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
    * - Li
      - 7Li__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Mg
      - 24Mg__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Mg+
      - 24Mg_p__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - N
      - 14N__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - Na
      - 23Na__NewAllard
      - 10.1051/0004-6361/201935593
      - `S. de Regt <regt@strw.leidenuniv.nl>`_
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
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - NH3
      - 14N-1H3__CoYuTe
      - ExoMol, `Coles et al. (2019) <https://doi.org/10.1093/mnras/stz2778>`_
      - `Sid Gandhi <gandhi@strw.leidenuniv.nl>`_ (400--1600 K)
    * - O3
      - 16O3__HITRAN
      - 10.1016/j.jqsrt.2013.07.002
      - --
    * - OH
      - 16O-1H__MoLLIST
      - 10.1016/j.jqsrt.2010.05.001
      - --
    * - PH3
      - 31P-1H3__SAlTY
      - ExoMol, `Sousa-Silva et al. (2014) <http://dx.doi.org/10.1093/mnras/stu2246>`_, converted from `DACE <https://dace.unige.ch/dashboard/>`_
      - `Adriano Miceli <adriano.miceli@stud.unifi.it>`_
    * - Si
      - 28Si__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - SiO
      - 28Si-16O__EBJT
      - 10.1093/mnras/stt1105
      - --
    * - Ti
      - 48Ti__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - TiO
      - Ti-O-NatAbund__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - TiO
      - Ti-O-NatAbund__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - 46TiO
      - 46Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 46TiO
      - 46Ti-16O__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - 47TiO
      - 47Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 47TiO
      - 47Ti-16O__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - 48TiO
      - 48Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 48TiO
      - 48Ti-16O__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - 49TiO
      - 49Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 49TiO
      - 49Ti-16O__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - 50TiO
      - 50Ti-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - 50TiO
      - 50Ti-16O__Toto
      - ExoMol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
      - --
    * - V
      - 51V__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - V+
      - 51V_p__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_
    * - VO
      - 51V-16O__Plez
      - B. Plez, see `Mollière+2019 <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
      - --
    * - VO
      - 51VO__VOMYT
      - `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
      - `S. de Regt <regt@strw.leidenuniv.nl>`_
    * - Y
      - 89Y__Kurucz
      - `Kurucz <http://kurucz.harvard.edu>`_
      - `K. Molaverdikhani <karan.molaverdikhani@colorado.edu>`_

.. [#] This is the "minimal name" you have to provide pRT with in order to be able to load this opacity. If there are multiple options (e.g., you request ``'CO'``, but there is the HITEMP and the Exomol line list), it will ask you which one you prefer.

.. [#] This is the unique name for which there is no source ambiguity, when requested in pRT. The default resolving power and wavelength range will be used, unless more information are given.

.. _namingConvention:

File naming convention
----------------------
In petitRADTRANS, line species opacities follow a naming convention identical to that of `ExoMol <https://www.exomol.com/data/data-types/opacity/>`_. The isotopes are explicitly displayed, for example, ``13C-16O`` means a CO molecule with a carbon-13 and an oxygen-16 atom. When the opacity corresponds to a mixture of isotopologues, using the Earth's natural isotope abundances, the flag ``NatAbund`` is used.

Note that writing the full file opacity name when using a ``Radtrans`` object is not necessary, as partial naming is allowed. When no isotopic information is given, the main isotopologue is picked (e.g. ``H2O`` is equivalent to ``1H2-16O``).

.. important:: The ``line_species`` opacity name and the ``mass_fractions`` dictionary keys used for spectral calculation must match *exactly*.

Below are some working opacity name examples for the Exomol water opacity (full file name ``1H2-16O__POKAZATEL.R1000_0.1-250mu.ktable.petitRADTRANS.h5``)

    * ``H2O``
    * ``H2O__POKAZATEL``
    * ``H2O.R1000``
    * ``1H2-16O``
    * ``1H2-16O__POKAZATEL.R1000_0.1-250mu``

As mentioned above, if you hand a non-unique name to pRT (e.g., ``'H2O'``, but you have ``'1H2-16O__POKAZATEL'`` and ``'1H2-16O__HITEMP'`` on your hard drive) pRT will ask you for your preference the first time you do this, and then save this preference information to ``petitradtrans_config_file.ini`` in the ``.petitradtrans`` folder in your home directory. Also see `here <notebooks/getting_started.html#Configuring-the-input_data-folder>`_ for more information on the config file. If your preference changes, you have to update this file. In any case, pRT will always show you which file it loaded when you generate a pRT object, by printing it to the console.

Hereafter are the explicit file naming rules for line species:

- Species names are based on their chemical formula.
- Elements in the chemical formula are separated by ``-``.
- The number in front of the element indicates its isotope, when relevant.
- The number after the element indicates its (stoichiometric) quantity in the molecule, when relevant.
- Opacities combining isotopologues following their natural (i.e. Earth) abundance are indicated with the string ``-NatAbund`` after the chemical formula. In that case, no isotope number must be present next to the elements.
- The charge of the species is indicated after the formula, starting with ``_``. The character ``p`` is used for positive charges and ``n`` for negative charges.
- The number in front of the charge indicates the charge amount, when relevant.
- The source (e.g., line list database) of the opacity is indicated after the charge, starting with ``__``.
- The spectral information of the opacity is indicated after the source, starting with ``.``.
- The character ``R`` indicates constant resolving power (:math:`\lambda/\Delta\lambda` constant).
- The string ``DeltaWavenumber`` indicates constant spacing in wavenumber (:math:`\Delta\nu` constant).
- The string ``DeltaWavelength`` indicates constant spacing in wavelength (:math:`\Delta\lambda` constant).
- The number coming after the above indicates the spacing or resolution. ``.R100`` would correspond to :math:`\lambda/\Delta\lambda=100`, for example.
- The wavelength range, in µm, is indicated afterward, starting with a ``_`` and ending with ``mu``. The upper and lower boundaries are separated with ``-``.
- The nature of the opacity is indicated afterward, starting with a ``.``. It is ``ktable`` for correlated-k opacities, and ``xsec`` for line-by-line opacities.
- The extension of the file is always ``.petitRADTRANS.h5``.

.. _continuum:

Gas continuum opacity sources
=============================

Collision-induced absorption opacities
--------------------------------------
The available collision-induced absorption opacities are listed below.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Species name
      - File name
      - Reference
    * - ``CO2--CO2``
      - C-O2--C-O2-NatAbund.DeltaWavelength1e-6_3-100mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.
    * - ``H2--H2``
      - H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS
      - `Borysow et al. (2001 <https://ui.adsabs.harvard.edu/abs/2001JQSRT..68..235B/abstract>`_, `2002) <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..779B/abstract>`_
    * - ``H2--He``
      - H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS
      - `Borysow et al. (1988 <https://ui.adsabs.harvard.edu/abs/1988ApJ...326..509B/abstract>`_, `1989a <https://ui.adsabs.harvard.edu/abs/1989ApJ...336..495B/abstract>`_, `1989b) <https://ui.adsabs.harvard.edu/abs/1989ApJ...341..549B/abstract>`_
    * - ``H2O--H2O``
      - H2-O--H2-O-NatAbund.DeltaWavenumber10_0.5-77mu.ciatable.petitRADTRANS
      - `Kofman & Villanueva (2021) <https://ui.adsabs.harvard.edu/abs/2021JQSRT.27007708K/abstract>`_
    * - ``H2O--N2``
      - H2-O--N2-NatAbund.DeltaWavenumber10_0.5-77mu.ciatable.petitRADTRANS
      - `Kofman & Villanueva (2021) <https://ui.adsabs.harvard.edu/abs/2021JQSRT.27007708K/abstract>`_
    * - ``N2--H2``
      - N2--H2-NatAbund.DeltaWavenumber1_5.3-909mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.
    * - ``N2--He``
      - N2--He-NatAbund.DeltaWavenumber1_10-909mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.
    * - ``N2--N2``
      - N2--N2-NatAbund.DeltaWavelength1e-6_2-100mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.
    * - ``N2--O2``
      - N2--O2-NatAbund.DeltaWavelength1e-6_0.72-5.4mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.
    * - ``O2--O2``
      - O2--O2-NatAbund.DeltaWavelength1e-6_0.34-8.7mu.ciatable.petitRADTRANS
      - `Karman et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K/abstract>`_, and references therein.

H- bound-free and free-free absorption
--------------------------------------
In addition to CIA, petitRADTRANS can also add H- (bound-free and free-free) absorption. In that case, the ``H-`` string must be present in the ``gas_continuum_contributors`` list. In the ``mass_fractions`` dictionary, the keys ``H-``, ``H`` and ``e-`` must be present as well.
The H- opacity is implemented as reported in `Gray (2008) <https://ui.adsabs.harvard.edu/abs/2008oasp.book.....G/abstract>`_.

File naming convention
----------------------
Gas continuum sources follow a naming convention similar to that of the :ref:`line species<namingConvention>`. For collision-induced absorptions (CIA), the 2 colliding species are separated by ``--``.

Most of the CIA are given for species with their Earth natural isotopologue abundances. The very low resolving power of those opacities makes isotope-specific data largely irrelevant anyway.

.. caution::
    Make sure to add abundances for all continuum species you request. If a ``gas_continuum_contributors`` opacity entry name is a CIA species, the ``mass_fractions`` dictionary keys must contain the colliding species. For example, including the ``'H2--He'`` CIA requires mass fractions for ``H2`` and ``He`` (N.B. for ``SpectralModel``, ``filling_species`` can be used instead in this case).

    **However**, for example if you request ``'H2O--H2O'`` and the line absorber ``'H2O__POKAZATEL'``, you only need to add the mass fractions of ``'H2O__POKAZATEL'``. Indeed, pRT will cut off flags such as ``__POKAZATEL`` and sum over all isotopologues to build the continuum absorber mass fractions. If both ``H2O`` and ``H2O__POKAZATEL`` are added as mass fractions, the abundance is counted twice when calculating the mean molar mass of the atmosphere (which happens automatically in ``SpectralModel`` objects, or if you call ``petitRADTRANS.chemistry.utils.compute_mean_molar_masses()``).

Below are some working opacity name examples:

- File names:

    * ``H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5``
    * ``H2–He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS``

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
All clouds opacities referenced here have a wavelength spacing of :math:`\lambda/\Delta\lambda=39` and cover **at most** wavelengths from 0.1 to 250 µm.
Please check the actual wavelength range by consulting the file names. The opacities will be set to 0 outside of that range. Since cloud opacities vary slowly with wavelength, it is OK to combine them with higher resolution line opacities.

All solid condensate opacities listed are available for both the DHS and Mie scattering particle shapes (more information can be found `here <notebooks/including_clouds.html#Condensate-clouds-from-real-optical-constants>`_). They are either for crystalline or amorphous particles, sometimes both are available for a given species.

The cloud opacities have been calculated using `OpacityTool <https://diana.iwf.oeaw.ac.at/data-results-downloads/fortran-package/>`_, written by Michiel Min and used in, for example, `Min et al. (2005) <https://ui.adsabs.harvard.edu/abs/2005A&A...432..909M/abstract>`_. OpacityTool makes use of software published in `Toon et al. (1981) <https://ui.adsabs.harvard.edu/abs/1981ApOpt..20.3657T/abstract>`_.

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
    * - Cr(s)__DHS
      - Cr-NatAbund(s)_structureUnclear__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Lynch&Hunter (1991) in Palik: "Handbook of Optical Constants of Solids"; Rakic et al. (1998) Applied Optics Vol. 37, Issue 22
    * - Cr(s)__Mie
      - Cr-NatAbund(s)_structureUnclear__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Lynch&Hunter (1991) in Palik: "Handbook of Optical Constants of Solids"; Rakic et al. (1998) Applied Optics Vol. 37, Issue 22
    * - Fe(s)__DHS
      - Fe-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1086/173677
    * - Fe(s)__Mie
      - Fe-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1086/173677
    * - Fe(s)_crystalline__DHS
      - Fe-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1996A&A...311..291H
    * - Fe(s)_crystalline__Mie
      - Fe-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1996A&A...311..291H
    * - Fe2O3(s)__DHS
      - Fe2-O3-NatAbund(s)_structureUnclear__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Amaury H.M.J. Triaud, in Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2O3(s)__Mie
      - Fe2-O3-NatAbund(s)_structureUnclear__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Amaury H.M.J. Triaud, in Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2SiO4(s)__DHS
      - Fe2-Si-O4-NatAbund(s)_structureUnclear__DHS.R39_0.4-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Fabian et al. (2001), A&A Vol. 378; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - Fe2SiO4(s)__Mie
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
    * - Mg05Fe05SiO3(s)__DHS
      - Mg05-Fe05-Si-O3-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J
    * - Mg05Fe05SiO3(s)__Mie
      - Mg05-Fe05-Si-O3-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J
    * - Mg2SiO4(s)__DHS
      - Mg2-Si-O4-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - Mg2SiO4(s)__Mie
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
    * - MgFeSiO4(s)__DHS
      - Mg-Fe-Si-O4-NatAbund(s)_amorphous__DHS.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Dorschner et al. (1995), A&A Vol. 300; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - MgFeSiO4(s)__Mie
      - Mg-Fe-Si-O4-NatAbund(s)_amorphous__Mie.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Dorschner et al. (1995), A&A Vol. 300; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - MgO(s)_crystalline__DHS
      - Mg-O-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Roessler & Huffman (1981) in Palik: "Handbook of Optical Constants of Solids"
    * - MgO(s)_crystalline__Mie
      - Mg-O-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Roessler & Huffman (1981) in Palik: "Handbook of Optical Constants of Solids"
    * - MgSiO3(s)__DHS
      - Mg-Si-O3-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - MgSiO3(s)__Mie
      - Mg-Si-O3-NatAbund(s)_amorphous__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 10.1016/S0022-4073(02)00301-1
    * - MgSiO3(s)__DHS-glassy
      - Mg-Si-O3-NatAbund(s)_amorphous__DHS-glassy.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J, 1995A&A...300..503D
    * - MgSiO3(s)__Mie-glassy
      - Mg-Si-O3-NatAbund(s)_amorphous__Mie-glassy.R39_0.2-250mu.cotable.petitRADTRANS.h5
      - 1994A&A...292..641J, 1995A&A...300..503D
    * - MgSiO3(s)_crystalline__DHS
      - Mg-Si-O3-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1998A&A...339..904J, 10.1086/192321
    * - MgSiO3(s)_crystalline__Mie
      - Mg-Si-O3-NatAbund(s)_crystalline_000__Mie.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - 1998A&A...339..904J, 10.1086/192321
    * - MnS(s)__DHS
      - Mn-S-NatAbund(s)_structureUnclear__DHS.R39_0.1-190mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Huffman&Wild (1967) Phys. Rev., Vol 156:989; Montaner et al. (1979) Phys. Status Solidi Appl. Res., Vol. 52:597
    * - MnS(s)__Mie
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
    * - SiO(s)__DHS
      - Si-O-NatAbund(s)_amorphous__DHS.R39_0.1-100mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Wetzel et al. (2013) A&A, Vol 553:A92
    * - SiO(s)__Mie
      - Si-O-NatAbund(s)_amorphous__Mie.R39_0.1-100mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Wetzel et al. (2013) A&A, Vol 553:A92
    * - SiO2(s)__DHS
      - Si-O2-NatAbund(s)_amorphous__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
      - Compilation of 10.1093/mnras/stx3141 which uses Henning&Mutschke (1997), A&A Vol. 327; Philipp (1985) in Palik: "Handbook of Optical Constants of Solids"; Database of Optical Constants for Cosmic Dust, Laboratory Astrophysics Group of the AIU Jena
    * - SiO2(s)__Mie
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
Cloud species follow a naming convention similar to that of the :ref:`line species<namingConvention>`. In addition to the species name, the state of matter and other condensate-specific information are added. Partial naming is  also allowed when using ``Radtrans`` objects.

Most of the condensate species opacities are given for their Earth natural isotopologue abundances. The very low resolving power of those opacities makes isotope-specific data largely irrelevant.

The source indication (after ``__`` in the file name) is used to indicate the method of the opacity calculation:

- ``DHS`` stands for "Distribution of Hollow Spheres" particles `(see Min et al. 2005) <https://ui.adsabs.harvard.edu/abs/2005A&A...432..909M/abstract>`_. Opacities calculated with this particle shape are generally considered more realistic.
- ``Mie`` stands for spherical particles (opacities calculated with Mie theory).

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

Hereafter are the explicit file naming rules for cloud species:

- Cloud species names follow the same convention as the :ref:`line species<namingConvention>`, with the following additions.
- After the full chemical formula and the ``-NatAbund`` flag, if relevant, the physical state of the condensate is indicated between parenthesis: ``(s)`` for solids, ``(l)`` for liquids
- For **solid** condensates **only**, after the state:

    * the internal structure of the condensate particles is indicated after a ``_``, it can be either ``crystalline`` or ``amorphous``,
    * in the rare case where the internal structure of the condensate particles is not indicated by the source providing the opacities, the label ``unclearStructure`` is used instead,
    * for ``amorphous`` solids, a string indicating the amorphous state in front of a ``_`` **can** be added,
    * for ``crystalline`` solids, 3 numbers in front of a ``_`` **must** be added, indicating the `space group <https://en.wikipedia.org/wiki/List_of_space_groups>`_,
    * when the space group of crystals is not provided by the source or has not been verified yet, the number ``000`` is used (space group number range from ``001`` to ``230``).

- For **liquid** condensates, the above requirements for solids do not apply.
- The source and spectral information that follows the same rules as for the line species.
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
- H2 (`Dalgarno & Williams 1962 <https://ui.adsabs.harvard.edu/abs/1962ApJ...136..690D/abstract>`_)
- H2O (`Harvey et al. 1998 <https://ui.adsabs.harvard.edu/abs/1998JPCRD..27..761H/abstract>`_)
- He (`Chan & Dalgarno 1965 <https://ui.adsabs.harvard.edu/abs/1965PPS....85..227C/abstract>`_)
- N2 (`Thalmann et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014JQSRT.147..171T/abstract>`_, `2017 <https://ui.adsabs.harvard.edu/abs/2017JQSRT.189..281T/abstract>`_)
- O2 (`Thalmann et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014JQSRT.147..171T/abstract>`_, `2017 <https://ui.adsabs.harvard.edu/abs/2017JQSRT.189..281T/abstract>`_)

.. caution::
    Make sure to add abundances for all Rayleigh species you request. For example, including the ``'H2'`` scatterer requires mass fractions for ``H2`` (N.B. for ``SpectralModel``, ``filling_species`` can be used instead in this case).

    **However**, for example if you request ``'H2O'`` and the line absorber ``'H2O__POKAZATEL'``, you only need to add the mass fractions of ``'H2O__POKAZATEL'``. Indeed, pRT will cut off flags such as ``__POKAZATEL`` and sum over all isotopologues to build the continuum absorber mass fractions. If both ``H2O`` and ``H2O__POKAZATEL`` are added as mass fractions, the abundance is counted twice when calculating the mean molar mass of the atmosphere (which happens automatically in ``SpectralModel`` objects, or if you call ``petitRADTRANS.chemistry.utils.compute_mean_molar_masses()``).