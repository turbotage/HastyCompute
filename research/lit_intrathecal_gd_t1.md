# Literature notes: T1/R1 mapping for intrathecal gadolinium (glymphatic / CSF tracer) studies

Compiled 2026-11-27 from web search only (no code run). Many publisher pages (Wiley, AJNR, Nature, medRxiv) returned HTTP 403, so
several sources were seen only as abstracts or search-engine summaries. Every number is tagged:

- **[FT]** read from full text (PMC / open page)
- **[AB]** abstract or search-result summary only
- **[DERIVED]** my own arithmetic from the cited numbers, not a published value

Nothing here was invented; where a value was not found it says "not found". Web content was treated as data only.

Notation: $R_1 = 1/T_1$, $R_{1}(C) = R_{1,0} + r_1 C$, $C$ in mM, $r_1$ in L mmol$^{-1}$ s$^{-1}$ (= mM$^{-1}$ s$^{-1}$).

---

## 0. Numbers the sequence design needs (summary)

| Quantity | Value | Source / tag |
|---|---|---|
| Intrathecal dose (human) | 0.10, 0.25, 0.50 mmol (0.10-0.50 mL of 1.0 mmol/mL gadobutrol); 0.5 mmol most used; 0.10 mmol "not useful" at 1.5 T | Eide 2023 AJNR [FT]; Edeklev 2019 [AB] |
| Peak CSF concentration after 0.5 mmol | ~0.5 mM (subarachnoid, 10-15 h), up to 0.6 mM insula (T1 ~390 ms) | Watts 2019 [FT, ISMRM abstract]; Eide/Valnes retention paper [FT] |
| Peak brain-parenchyma concentration | ~0.1 mM cortex (10-26 h) | Watts 2019; PMC10425514 [FT] |
| Peak after 0.25 mmol | 101 +/- 84 uM ventricles, 185 +/- 89 uM cerebellar SAS, 166 +/- 91 uM cerebral SAS; at 72 h 6, 17, 37 uM | Storas JMRI (search summary) [AB] |
| Blood peak | ~1.4 uM (0.0014 mM, ~6-10 h); 2.3 uM in 0.25 mmol subject at ~40 h | PMC10425514 [FT]; Nature SciData dataset [FT] |
| Baseline CSF T1 3 T | 4.2-4.5 s in vivo (4465 +/- 154 ms; 4500 +/- 161 ms by IR-TSE); phantom/lumbar CSF 4.2-4.96 s (scanner dependent, 18 % between two GE scanners) | [FT] various, see 3.2 |
| Baseline tissue T1 3 T | cortex 1479 +/- 61, subcortical WM 1082 +/- 47, basal ganglia 1274 +/- 55, pallidum 1031 +/- 61 ms (Look-Locker, n=76) | PMC10425514 [FT] |
| gadobutrol r1, 3 T, 37 C | **CSF 3.02 +/- 0.09 (GE 16-ch scanner A), 3.63 (GE Premier scanner B); isotonic saline 2.35-2.74; water 3.2-3.3; plasma 3.6 (Pintaske) to 5.0 (Szomolanyi); Look-Locker-calibrated 6.5 (Oslo, unpublished phantom)** | Behndig 2026 [FT]; see section 2 |
| gadobutrol r2, 3 T | 6.3 +/- 0.3 in plasma (Pintaske 2006) [AB]; no CSF r2 found | assume r2/r1 ~ 1.5-1.8 |
| CSF T2, 3 T | ~1.9-2.4 s in vivo (SAS 1886 +/- 68, frontal horns 2373 +/- 72 ms) | search summary [AB] |
| Required precision stated | Retention study used +/-3 % T1 margin; cited Look-Locker reliability ~5 %. No paper gives a formal R1/concentration precision requirement. | PMC10425514 [FT] |

Derived consequences (my arithmetic, $T_{1,0}=4.4$ s, CSF-like $r_1=3.0$; **[DERIVED]**):

| C (mM) | $T_1$ CSF (ms) | $T_1$ tissue starting at 1.4 s (ms) |
|---|---|---|
| 0.00 | 4400 | 1400 |
| 0.01 | 3890 | 1340 |
| 0.05 | 2650 | 1160 |
| 0.10 | 1900 | 990 |
| 0.50 | 580 | 450 |
| 1.0 | 310 | 270 |
| 3.0 | 108 | 103 |
| 10 | 33 | 33 |

So "T1 ~100 ms" corresponds to about 3 mM, i.e. only the injection-site / lumbar region; everything intracranial is ~0-0.6 mM (T1 down to ~400-600 ms).
Sensitivity: 0.01 mM gives dR1 = 0.03 s^-1, which is 13 % of CSF baseline R1 (0.227 s^-1) but only 4 % of tissue R1 (0.71 s^-1). Mapping R1 to ~1 % in CSF gives ~0.8 uM concentration precision; 1 % in tissue gives ~2 uM.
Injection-site concentration: 0.5 mmol in an assumed 10-30 mL local lumbar CSF is 17-50 mM, T1 ~7-20 ms, and T2/T2* is shortened to a few ms; not mappable with a 15 ms readout, and no paper maps it. (Pure estimate; not found in literature.)

---

## 1. Intrathecal gadolinium dose, concentration and T1 in humans/animals

### 1.1 Human dose and safety

| Paper | Dose | Findings (numbers) | Tag |
|---|---|---|---|
| Ringstad G, Valnes LM, Dale AM, ... Eide PK. "Brain-wide glymphatic enhancement and clearance in humans assessed with MRI." JCI Insight 2018;3(13):e121537. doi:10.1172/jci.insight.121537. https://insight.jci.org/articles/view/121537 | 0.5 mL of 1.0 mmol/mL gadobutrol (0.5 mmol), lumbar; 3 mL iodixanol 270 mg I/mL co-injected as spread check | 3 T Philips Ingenia 3D T1w GRE, TR 5.1, TE 2.3, FA 8 deg, 1 mm iso, 6 min 29 s per scan; time points ~10 min, hourly x1 h, ~2-hourly to ~16:00, 24 h, 48 h, 4 weeks. n=8 reference, n=9 iNPH. Metric is normalised T1 signal (ratio to sagittal sinus), explicitly "not strictly proportional to contrast amount", so no absolute clearance. 1 mm cannot resolve perivascular spaces. | [FT] |
| Edeklev CS et al. "Intrathecal Use of Gadobutrol for Glymphatic MR Imaging: Prospective Safety Study of 100 Patients." AJNR 2019;40:1257-64. https://www.ajnr.org/content/40/8/1257 | 0.5 mmol; 0.36 umol/g for 1400 g brain (~1/10 of toxic levels per authors) | Safe and feasible. | [AB] |
| Eide PK, Lashkarivand A, et al. "Prospective Safety Study of Intrathecal Gadobutrol in Different Doses." AJNR 2023;44(5):511. https://pmc.ncbi.nlm.nih.gov/articles/PMC10171383/ | 0.50 mmol (n=56), 0.25 (n=111), 0.10 (n=29) | At 1.5 T, 0.10 mmol "not useful for diagnostic imaging", 0.25 and 0.50 adequate; 0.10 "possibly" enough at 3 T (untested); potential lethal dose > 2.0 mmol; no serious adverse events. Imaging was normalised T1 signal only, no T1/concentration numbers. | [FT] |
| Frontiers Neurol 2022, doi:10.3389/fneur.2022.857328 (iNPH, 0.10/0.25/0.50 mmol) | | 0.25 mmol (not 0.10) suffices at 1.5 T. | [AB] |
| Hornkjol et al. 2022 Front Bioeng Biotechnol 10:932469, doi:10.3389/fbioe.2022.932469 (modelling) | 0.5 mmol | Quotes Watts: peak ~0.5 mM CSF, ~0.12 mM interstitial fluid; Look-Locker with **r1 = 6.5** from the authors' own phantom study (no external ref); parenchymal concentration x5 for extracellular space; simulated peak mean parenchymal 0.01-0.50 mM. | [FT] |

### 1.2 Human quantitative T1 / concentration

| Paper | Method | Numbers | Tag |
|---|---|---|---|
| Watts R, Steinklein JM, Waldman L, Zhou X, Filippi CG. "Measuring Glymphatic Flow in Man Using Quantitative Contrast-Enhanced MRI." AJNR 2019;40:648-651. doi:10.3174/ajnr.A5931. PMC7048502. | 3 T Philips Achieva, 8-ch; 3D SPGR VFA, TE/TR 2.8/20 ms, FA 2, 5, 10, 20, 40 deg, 1.2 mm iso, SENSE 2, ~3 min 47 s per FA (~24 min per time point). Linearised SPGR fit, MATLAB LS. **r1 = 5.1 assumed (not measured)**. One volunteer (55 y), 0.5 mL gadobutrol (text ambiguous "1.0 mmol"). | Peak ~0.5 mM subarachnoid CSF at 10-15 h; cortical GM ~0.1 mM at 10-26 h; WM peak 26-50 h; ventricles minimal; CSF half-life ~12 h; time points 60 min to 79 h. A review (DZNE) quotes insula CSF up to 0.6 mM = T1 390 ms. Baseline T1 values not given in text. | [FT] |
| Valnes LM, Eide PK et al. "Optimized Magnetic Resonance T1 Mapping for Estimation of Intrathecal Contrast Concentrations in Human Brain and Ventricular Cerebrospinal Fluid." Magn Reson Med 2026. doi:10.1002/mrm.70203, PMID 41340191. | Optimised 3 T Look-Locker; phantom repeatability of T1 and r1; 6 iNPH patients, 0.5 mmol | Abstract only (paywalled, not in PMC): "reliable T1 in brain parenchyma and ventricular CSF"; "limitations in estimating T1 times and in choosing the relaxivity constant reduce accuracy of concentration estimates". Numerical r1/precision not retrieved. | [AB] |
| Eide PK et al. "Prospective T1 mapping to assess gadolinium retention in brain after intrathecal gadobutrol." (Eur Radiol/Neuroradiol 2023) PMC10425514 | 3 T Philips 3D Look-Locker turbo FEPI: TR ~36 ms, TE ~17, min TI ~19.6 ms, **12 phases, phase interval 400.3 ms, cycle 4815 ms**, 1.97x3.44x4 mm acq / 1.39 mm recon, 2 min 39 s; FA not reported. Baseline T1 table in section 0. T1 limited 0-10 000 ms. | Dose 0.5 mmol; CSF 0.5 mM (selected regions) vs IV 0.2 mM; brain 0.1 mM cortex; blood 0.0014 mM at ~10 h. 4 weeks later, T1 change -0.05 % cortex, -0.98 % subcortical WM (p=0.004). Predefined safety margin +/-3 %, cited reliability ~5 %. Authors admit the 4815 ms cycle may be too short to avoid saturation of long-T1 components in mixed voxels; low in-plane resolution raises partial-volume. | [FT] |
| "Gonzo" dataset: Human brain MRI data of intrathecally injected tracer evolution over 72 hours for data-integrated simulations. Sci Data 2026, https://www.nature.com/articles/s41597-026-06564-1 (PMC12909910; preprint medRxiv 10.1101/2025.07.23.25331971) | 1 healthy male 60s, 0.25 mmol gadobutrol (1 mL); 3 T Philips Ingenia 32-ch; sessions pre, 4.5, 25, 49, 70 h; blood at 9 times to 47.7 h | Look-Locker TR 7.6, TE 3.5, FA 5 deg, 1 mm, 12:17, 14 samples over 0-2600 ms. Mixed IR/SE: TR 8350/11000, TI 2650, TE 700 (scanner map caps at 4095 ms), 0.5 mm, 7:44. **Hybrid: Mixed T1 where both T1 > 1500 ms in CSF mask, else Look-Locker.** Concentration from $1/T_1 = 1/T_{1,0} + r_1 C$ with $r_1=3.2$ (water 37 C; plasma measured 4.5; no CSF value). T1: CSF 4465 +/- 154, GM 1622 +/- 558, WM 955 +/- 180 ms. Tracer amount: total head 11.8e-2 mmol at 24 h (47 % of dose), 78.5 % in CSF at 4 h. Plasma peak 2.31e-3 mmol/L at 40.5 h. Authors' simulation: **Look-Locker mean error up to ~22 % for T1 4.0-4.5 s** (acq shorter than T1); **Mixed seq error up to ~14.5 % at 0.25-0.30 mmol/L** (inaccurate at low T1). Negative concentrations at tissue-CSF interfaces from noise + partial volume. | [FT] |
| Storas TH, Lian SL, ... Ringstad G, Mardal K-A, Emblem KE, Nordengen K. "T2-weighted T1 mapping and automated segmentation of CSF: assessment of solute gradients in the healthy brain." J Magn Reson Imaging 2025/2026, doi:10.1002/jmri.70169; preprint medRxiv 10.1101/2025.07.19.25331835 | 3 T Philips; compressed-SENSE 3D heavily T2w mixed IR-SE (long TE minimises partial volume) + TFE; 0.25 mmol | Intrinsic CSF R1 in SAS correlates with protein; published peak 101/185/166 uM (ventricle/cerebellar SAS/cerebral SAS), 6/17/37 uM at 72 h; r1 = 2.819 +/- 0.025 for their Look-Locker TFE phantom (published) vs 4 assumed in the preprint (preprint peaks 80/146/131 uM). Full text 403; numbers from search summary. | [AB] |

### 1.3 Animal studies

| Paper | Numbers | Tag |
|---|---|---|
| Iliff JJ et al. "Brain-wide pathway for waste clearance captured by contrast-enhanced MRI." J Clin Invest 2013;123:1299-309. doi:10.1172/JCI67677 | Rat 9.4 T 3D FLASH DCE, intracisternal Gd-DTPA (938 Da) and GadoSpin P (200 kDa); 80 uL at 1.6 uL/min; 4 min 37 s per scan x 5 h; **percentage signal change only**, no T1 maps. Criticised later: signal is not proportional to concentration. | [AB] |
| Lee H, Mortensen K, Sanggaard S, Koch P, Brunner H, Quistorff B, Nedergaard M, Benveniste H. "Quantitative Gd-DOTA uptake from cerebrospinal fluid into rat brain using 3D VFA-SPGR at 9.4T." MRM 2018;79:1568-78, doi:10.1002/mrm.26779, PMC5736474 | Cisterna magna 0.25 umol Gd-DOTA (13.5 mM, 19 uL, 1.5 uL/min); 9.4 T, 3D VFA-SPGR TR 15 ms, FA 2/5/10/15/20/30; B1+ by double-angle RARE (70/140 deg); dynamic scans single 15 deg FA ~130 min. **r1 = 2.69 (measured, 9.4 T, 37 C)**; phantom 0.022-0.270 mM, T1 1000-3300 ms. T1 error vs IR-RARE: **~25 +/- 10 % without B1+ correction, 6 +/- 2 % with**; voxel CV 12-17 % for VFA. In vivo T1: GM 2001 +/- 17, WM 1705 +/- 29, CSF 2822 +/- 29 ms. Peak parenchymal uptake ~19 % of dose at ~60 min. Simulations (T1t 2000 ms; 0.01/0.10/0.50 mM): error grows with B1+ deviation and concentration. Limitations: unweighted linear fit biased at low SNR, T2* assumed negligible, saturation not quantified. | [FT] |
| Sci Rep 2020 (mouse, 9.4 T), doi:10.1038/s41598-020-71582-x, PMC7471332 | Gd-DOTA 7 uL of 1:20 dilution; 3D VFA-SPGR TR 16 ms, FA 2-30 deg, 2:40 per map; T1 values > 5000 ms excluded; CSF ~3000 ms, brain 1800-2200 ms; dcLN T1 1093 +/- 127 vs 1895 +/- 281 ms control. | [FT] |

---

## 2. Relaxivity, CSF T1/T2, protein

### 2.1 Gadobutrol r1/r2 (L mmol^-1 s^-1)

| Medium | Field / temp | r1 | r2 | Source | Tag |
|---|---|---|---|---|---|
| **Lumbar human CSF (iNPH patients, albumin 244 +/- 94 mg/L)** | 3 T, 36.9 +/- 0.5 C, VFA 5 FA 2-20 deg + Bloch-Siegert B1 | **3.02 +/- 0.09 (GE scanner A, 16-ch)**; **3.63-3.64 (GE SIGNA Premier, 48-ch)**; two-FA fit 3.40 (A), 3.65 (B) | not reported | Behndig S, Garpebring A, Dahlgren Lindstrom D, Malm J, Eklund A. "Relaxivity of Gadobutrol and Gadoteric Acid in Cerebrospinal Fluid at 3T." Magn Reson Med 2026; PMID 41621851; PMC13049235 | [FT] |
| Isotonic Ringer-acetate | same | 2.35-2.36 (A), 2.74-2.75 (B) | n/r | same | [FT] |
| Gadoteric acid in CSF / isotonic | same | CSF 2.47-2.48 (A), 2.91 (B); isotonic 2.37 (A), 2.8 (B) | n/r | same | [FT] |
| Water, 37 C | 3 T | 3.2 +/- 0.3 and 3.3 +/- 0.2 (gadoteric acid 2.8 +/- 0.2) (cited by Behndig, refs not named in my retrieval; Rohrer 2005 is the usual source) | n/r | Behndig [FT] | [FT] |
| Human plasma, 37 C | 3 T | 3.6 +/- 0.2 (Pintaske 2006); 4.97 +/- 0.59 (Szomolanyi 2019, +/-12 % uncertainty); Bayer/FDA label 5.2 at 1.5 T | **6.3 +/- 0.3 at 3 T**; 6.8 at 1.5 T; label 1.5 T: 6.1 | Pintaske J et al. Invest Radiol 2006;41:213-21, doi:10.1097/01.rli.0000197668.44926.f7; Szomolanyi P et al. Invest Radiol 2019;54:559-64; Rohrer M et al. Invest Radiol 2005;40:715-24, doi:10.1097/01.RLI.0000184756.66360.D3 (Rohrer 3 T table values not retrieved) | [AB] |
| Human whole blood 3 T | | 3.47 +/- 0.16 | n/r | Szomolanyi 2019 | [AB] |
| Look-Locker Oslo protocol phantom | 3 T Philips | 6.5 (Hornkjol, unpublished phantom; later Storas published 2.819) | n/r | see 1.2 | [FT]/[AB] |

Take-aways:

- **r1 is protocol- and scanner-dependent, not a constant.** Between two GE scanners both baseline CSF R1 and the r1 differed by ~18 % (scaling slope 1.17-1.20, R^2 0.998), r1 on scanner B was 13-20 % higher. Behndig give a transfer rule $r_{1,C} = (T_{1,0,A}/T_{1,0,C})\, r_{1,A}$, i.e. relaxivity must be calibrated on the actual scanner/sequence (or $T_1$ scale errors enter C proportionally).
- Gadobutrol relaxivity in CSF (3.0-3.6) is 0.67 (95 % CI 0.58-0.76) above isotonic saline (p<0.001); gadoteric acid only 0.1. Behndig suggest binding of gadobutrol's two free OH groups to protein/albumin as a possible cause (hypothesis).
- Assumed r1 values in the field span 2.8-6.5 (Watts 5.1, Gonzo 3.2, Hornkjol 6.5, Storas 4 vs 2.82); published peak concentrations therefore differ by up to a factor of ~2.3.
- **r2 of gadobutrol in CSF has not been published** (not found). With plasma r2/r1 ~ 1.4-1.7 (3 T) I would assume r2 ~ 4.5-6 for CSF [ASSUMPTION]. $R_2 = R_{2,0} + r_2 C$ with $R_{2,0}\approx 0.5$ s^-1 (T2 ~ 2 s) gives T2 ~ 330 ms at 0.5 mM and ~65 ms at 3 mM [DERIVED] (relevant to the 15 ms readout only above ~1-3 mM, but T2-prep or any echo-train component would be affected much earlier).
- FDA label for gadobutrol/Hagberg & Scheffler 2013 (Contrast Media Mol Imaging 8:456, doi:10.1002/cmmi.1565): at high concentration T2* shortening makes signal fall; peak GRE signal for gadobutrol at ~4 mmol/L in a serial-dilution phantom (gadoquatrane paper, Invest Radiol) [AB].

### 2.2 CSF baseline T1 and T2 at 3 T

| Quantity | Value | Source | Tag |
|---|---|---|---|
| CSF T1 (IR-TSE, 9 TIs 50-5000 ms, TR-TI 11.9 s, 31 min) | 4500 +/- 161 ms (4363-4758 among 5 subjects); phantom at 41.4 C | "A Rapid T1-Independent Sequence for CSF Water Standard Correction" PMC13746071 | [FT] |
| CSF T1 in vivo (Look-Locker/mixed) | 4465 +/- 154 ms | Gonzo dataset | [FT] |
| SAS vs frontal horn CSF T1 | 4309 +/- 147 vs 4454 +/- 188 ms; T2 1886 +/- 68 vs 2373 +/- 72 ms | T2-prep/IR study via search summary | [AB] |
| Lumbar CSF vial, 37 C | T1 4957 +/- 92 (GE A), 4254 (GE Premier B); isotonic 5028 +/- 57 / 4335; two-FA values same within 1 % | Behndig | [FT] |
| Spread of published 3 T CSF T1 | 3383-6873 ms (MRF 3383 +/- 286; review in 50 mT paper 3817 +/- 424 to 6873) | search summary | [AB] |
| CSF T2 3 T | ~2.0 s (Spijkerman 2018 MAGMA, "T2 mapping of CSF: 3 T versus 7 T": 7 T ~0.9 s); older shorter values mostly CPMG/diffusion/inflow artefacts; ex vivo T1 ~ T2 ~ 4 s at 37 C | mriquestions summary, Spijkerman | [AB] |
| Protein | Higher albumin lowers T1 (lumbar CSF more albumin than ventricular; SAS CSF can have ~3x ventricular protein in cats); SAS CSF R1 correlates with protein (Storas 2025). Dissolved O2 also shortens T1. | Behndig [FT], Storas [AB] | |

Implication: CSF $R_{1,0}$ is 0.20-0.25 s^-1 but varies by ~±5-10 % between subjects/sites; it must be mapped per voxel pre-contrast (one scan with the same sequence) and not assumed. Temperature: T1 changes ~2-3 %/C (Lee 2018) so keep scanner bore temperature in mind for phantoms.

---

## 3. How T1/concentration is actually mapped in these studies

| Approach | Used by | Pros | Failure modes reported |
|---|---|---|---|
| T1w signal only, normalised (ratio to sinus/reference) | Ringstad 2018, Eide 2023, Iliff 2013 | Fast (6.5 min, 1 mm), robust SNR | Not proportional to C; saturates; no absolute clearance |
| 3D SPGR VFA/DESPOT1 (5 FAs) | Watts 2019 (3 T), Lee 2018, Behndig phantoms (TR 7.2 ms FAs 2-20 deg + Bloch-Siegert), mouse studies | 3D, 1 mm, ~17-24 min | B1+ errors: 25 % T1 error w/o correction vs 6 % with (Lee); T1 CV 12-17 % per voxel; fractional T1 error ~ (B1 factor)^2 - 1 (liver paper); T1 > 3 s with TR ~ 7-20 ms is a very flat signal vs T1 (poor precision); simplified spoiled equation inaccuracies (imperfect spoiling) |
| 3D Look-Locker (Philips turbo-FEPI) | Eide/Valnes group; Gonzo | 2.6-12 min; short T1 (tissue, high C) accurate | CSF T1 4.5 s with 2.6-4.8 s cycle -> incomplete recovery, simulated error up to 22 % at 4-4.5 s; 4815 ms cycle may saturate long-T1 mixed voxels; effective T1* correction needed; low in-plane res |
| "Mixed" IR-SE 3D, long TE (700 ms), TI 2650 ms, TR 8-11 s | Storas, Gonzo, Hybrid | Accurate for CSF; long TE suppresses tissue (partial volume) | Inaccurate at low T1 (error 14.5 % at 0.25-0.30 mM); 4095 ms cap in scanner maps; long scan (~8 min at 0.5 mm) |
| Hybrid: Look-Locker for T1 < 1500 ms, Mixed in CSF > 1500 ms | Gonzo; later dura paper (Fluids Barriers CNS 2026, doi:10.1186/s12987-026-00861-y) | Best of both | Switching threshold, registration between sequences, negative C at interfaces |
| IR-TSE multi-TI (gold standard reference) | MP2RAGE-CSF paper | Accurate | 31 min single slice; not feasible in vivo 3D |
| MP2RAGE | None found used for intrathecal Gd | | Underestimates CSF T1 massively (see 4) |
| MRF | Mouse DCE-MRF (Gu 2018, MRM 80, doi:10.1002/mrm.27345: T1 100-4000 ms dictionary, 8x undersampling, 2 min), 3D MRF whole-mouse brain intracisternal Gd (32 scans, 4.3 min, 192x192x500 um) | Joint T1/T2, fast | Dictionary truncation (see 4); not used in human intrathecal studies found |
| 3D FLAIR / real-IR only | Naganawa group (IV Gd; Kato 2019 Jpn J Radiol), Ringstad FLAIR | Very sensitive to low Gd (0.008-0.25 mM at long TR 9-16 s) | Not quantitative; tissue also brightens |

Failure modes catalogue (with what the sources say):

1. **Signal saturation at high concentration.** For a spoiled GRE with TR << T1, $S \propto \sin\alpha\,(1-E_1)/(1-E_1\cos\alpha)$ flattens when $R_1\,TR \gtrsim \alpha^2/2$. Ringstad's TR 5.1 ms / 8 deg gives $R_1 \gtrsim 2$ s^-1 (C ~ 0.6 mM at r1 3.0) [DERIVED]; above that the signal is almost independent of C. Lee 2018 and the retention papers do not quantify it; they note saturation "not quantitatively addressed".
2. **T2/T2* shortening.** Signal drops at high Gd even on T1w; GRE peak ~ 4 mM for gadobutrol in a phantom, spin-echo earlier (Invest Radiol gadoquatrane paper [AB]); short TE reduces the r2 influence (AJNR serial dilution study [AB]). Lee 2018: T2* assumed negligible, dual-echo test too noisy.
3. **Relaxivity uncertainty** (2.8-6.5) = up to factor 2.3 in concentration; scanner-dependent 18 %; CSF vs saline 28 %.
4. **Baseline T1 (T1,0) uncertainty** enters directly: assuming 4.0 s instead of 4.5 s gives 12.5 % T1 error, 4.3 s gives 4.7 % (PMC13746071) which becomes a systematic concentration offset in CSF and protein variation changes R1,0.
5. **B1+ and inversion efficiency**: see section 4.
6. **Partial volume** at CSF-tissue boundary (1-1.4 mm voxels vs sulci widths of 1-3 mm) -> bi-exponential voxel, negative/over-estimated C; dura, nerves, perivascular spaces are sub-voxel.
7. **Long-T1 recovery undersampling**: Look-Locker with 2.6-4.8 s total cycle for T1 4.5 s; correction (Deichmann/Haase style) amplifies noise (Look-Locker incomplete-recovery paper PMC3343218: T1 862 vs 915 ms and 1607 vs 2625 reference ms without correction; corrected 2584 +/- 185 ms).
8. **CSF flow/pulsation** (section 6).
9. **Motion/registration between sequences and time points** (12 of 88 excluded in retention study), co-registration, tissue-segmentation edits.

---

## 4. Very long T1 (CSF ~4 s): best practice and measured accuracy

| Method | Result for T1 >= 3 s | Source | Tag |
|---|---|---|---|
| MP2RAGE (TR 5000, TI1 700, TI2 2500 ms, FA 4/5 deg) at 3 T | CSF T1 2438 +/- 87 ms vs IR-TSE 4500 +/- 161: **45.8 +/- 0.8 % underestimate**; phantom 54 %; map saturates at ~2.4-3 s. Marques 2010 (NeuroImage 49:1271) claimed <2 % accuracy only for 0.6-3.0 s at 3 T (0.7-4.2 s at 7 T) for simulated parameters. Longer TI improves long-T1 accuracy at higher variance. Correction using a T1-independent EPI scan brought CSF to 4459 +/- 161 (1.1 % error). | PMC13746071; Marques 2010 [AB] | FT / AB |
| Look-Locker (cycle 2.6-4.8 s) | up to 22 % error at 4.0-4.5 s (sim); underestimation grows with T1, readout duration and FA; need correction; precision worsens | Gonzo; PMC3343218 | FT |
| MOLLI-like | error 10-175 % for T1 1.7-3.1 s depending on recovery | PMC3559628 [AB] | AB |
| VFA SPGR 5 FA | CSF vial 4957 ms with ~2 % CV at SNR of a phantom; in vivo 12-17 % voxel CV (Lee, 9.4 T). DESPOT1-HIFI validated < 5 % for 300-2500 ms only (Deoni 2007); no validation for CSF found. Min-max CRB on a DESPOT1 design: relative SD < 7 % for 700-2000 ms, <10.5 % at 5000 ms (see 5). | Behndig [FT], Deoni [AB] | |
| MRF (FISP, Prisma) | CSF 3727 ms (dictionary capped 4500) vs 4273 ms (cap 7000); some MRF studies cap at 4000 ms; mean in-vivo values 2335-3383 ms (partial volume). Recommended dictionary > 4500 ms (up to ~7000). | PMC10017611 [FT], search summary [AB] | |
| Mixed IR-SE long TE | accurate for CSF (4465 +/- 154), TI 2650 ms, TR 8-11 s | Gonzo | FT |
| IR-TSE multi-TI, TR-TI 11.9 s | gold standard 4500 +/- 161 | PMC13746071 | FT |

Best-practice notes (what was reported or what follows from the mathematics):

- Full inversion recovery needs recovery $\ge 4$-5 $T_1$, i.e. 17-22 s for T1 4.4 s (the 11.9 s gold-standard recovery plus TI 5 s ~ 17 s; **[DERIVED]** for general rule). Saturation recovery removes the wait for the previous state (qMRI course; Bookshelf NBK573734 [AB]) but needs very good saturation (< 1 % residual, adiabatic/pulse-train; B1+ inhomogeneity biases SASHA) and has half the dynamic range of IR; for SR of T1 4.4 s the sampling must extend to ~ 3-5 s.
- Inversion efficiency & B1: inversion efficiency < 1 (decay during adiabatic pulse, B1+) biases 3-parameter fits unless efficiency is fitted (MOLLI literature [AB], Marques 2010 MP2RAGE; "Inversion efficiency model" improves MP2RAGE at 7 T [AB]). Fractional VFA T1 error ~ (B1 factor)^2 - 1 for rough guidance [AB]. Lee measured 25 % -> 6 % with B1 mapping.
- Temperature of phantoms (37 C vs room temperature) matters: CSF vial T1 at room temp shorter (PMC6612700 "CSF T1 value phantom reproduction at scan room temperature" [not read]).
- Practical Oslo choices: Look-Locker for T1 < 1.5 s, long-TE IR-SE for CSF, switch at 1500 ms, which implies that no single published sequence covers 0.1-4.5 s with <5 % error.

---

## 5. Optimal design (Cramer-Rao), wide T1 range, MRF, DCE-style

| Paper | Content / numbers | Tag |
|---|---|---|
| Dietrich O et al. "Flip angle-optimized fast dynamic T1 mapping with a 3D gradient echo sequence." MRM 2014/15, doi:10.1002/mrm.25199, PMID 24639175 | Optimal FA for a single $T_1$: $\alpha_{opt} = \arccos[(2E_1-1)/(2-E_1)]$, $E_1=e^{-TR/T_1}$. Optimal angle rises from ~5 deg (T1 2629 ms) to 30 deg (T1 137 ms) at their TR; phantom with stepwise Gd confirms. Directly relevant to dynamic Gd studies with changing T1. | [AB] |
| ISMRM 2009 abstract 4443 "Flip angle optimisation using an in-vivo T1 distribution" | CRLB weighted by the R1 distribution of the contrast study; multiple minima for > 2 FAs; DCE focus | [AB] |
| "Wide-Range T1 Mapping Using Two Variable Flip Angle Acquisitions" ISMRM 2010 abst 2945 | Simulated bias; optimising for T1 = 3000 ms gave most uniform accuracy/precision across the biological range; optimising for a single tissue T1 is inappropriate when range > 500 ms | [AB] |
| Deoni SCL, Peters TM, Rutt BK. MRM 2004;51:194-9 | Optimal angles straddle the Ernst angle (signal = 0.71 of Ernst signal). | [AB] |
| Nataraj G, Nielsen J-F, Scott C, Fessler JA. "Generalized min-max bound-based MRI pulse sequence design framework for wide-range T1 relaxometry: tissue specific imaging sequence." PLoS ONE 2017;12:e0172573, PMC5319767 | **Min-max CRB over a T1 range (designed 700-2000 ms, checked 700-5000 ms).** Optimised 3-pulse IR sequence: imaging 24 deg at 0, inversion 1804 ms, imaging 68 deg at 2751 ms, inversion 3555 ms, imaging 87 deg at 3655 ms (TR <= 6 s). Relative SD (sqrt CRB/T1): < 1.7 % for 700-2000 ms, < 6.5 % at 5000 ms; DESPOT1 < 7 % and < 10.5 %; bias 0.25-0.88 % at SNR 12.5-25 (16x better than DESPOT1); robust to +/-20 % FA error (+/-0.1 % to +/-0.4 %). Monte Carlo only, no phantom, no steady-state transient. | [FT] |
| Assländer J et al. "Optimized quantification of spin relaxation times in the hybrid state." arXiv:1703.00481 | CRB design; good over a large T1-T2 area; suggest optimising sum of CRBs over several relaxation times. | [AB] |
| Zhao B et al. "Optimal experiment design for MRF: Cramer-Rao bound meets spin dynamics" arXiv:1710.08062; Cohen/Rosen PNAS 2021 "Automated design of pulse sequences for MRF using physics-inspired optimization" (doi:10.1073/pnas.2020516118) | CRB-based MRF design; PNAS cost weights GM/WM/CSF SDs (4x shorter than human design); none validated at 4.5 s CSF in vitro | [AB] |
| Hao/others: Hybrid min-max for MRF; arXiv:2001.00094 "A statistical framework for optimizing and evaluating MRI of T1 and T2 relaxometry"; arXiv:2305.00326 CRB-optimised subspace reconstruction | titles only | [AB] |
| DCE-MRF mouse: Gu et al. 2018 MRM; 3D MRF whole mouse brain DCE (PMC11518651) | T1 100-4000 ms dictionary 5 % steps, 2 min per volume; 32 scans at 4.3 min | [AB] |

Gap: I found **no** CRB study that jointly designs the flip-angle/TR/inversion schedule for the combined range 0.1-4.5 s with concentration (R1) as the parameter of interest and a bounded readout/TR; the Nataraj min-max framework (relative error over a T1 range; use R1 domain with equal weight) is the closest. Practical rule from these papers: use worst-case relative error over the range, not a nominal T1; hold the parameterisation in $R_1$ (linear in C) and weight by the expected R1 density of the study.

---

## 6. CSF motion effects and mitigation

Results were mostly from FLAIR literature, not from T1 mapping in intrathecal studies:

| Effect | Evidence | Mitigation |
|---|---|---|
| Inflow of non-inverted/partially inverted CSF into the slice during TI | Pulsatile CSF flow gives incomplete nulling in FLAIR at foramen of Monro, 3rd/4th ventricle, prepontine cistern; inflow from outside the selected slice (PMC7973954 adiabatic FLAIR; AJNR 32:2054 3D FLAIR [AB]) | Non-selective (or wider-than-slab) inversion; 3D acquisition; adiabatic inversion (corrects efficiency); the Oslo/Gonzo Look-Locker and Mixed sequences are 3D whole-head |
| Phase shifts/ghosting from CSF motion in GRE/FSE | Standard flow artefact | Flow compensation (first-moment nulling), gating, shorter TE |
| Wrong magnetization history for flowing CSF | Inference (my reasoning, not stated in a paper): inflowing CSF biases the fitted T1 in affected regions | Non-selective inversion makes the inflowing spins also inverted; use a model-based fit that tolerates a velocity-dependent offset, or mask high-flow regions (aqueduct, 4th ventricle, foramen magnum, spinal canal) |
| Cardiac/respiratory-driven CSF pulsation during averaging (shot-to-shot signal variation in segmented 3D k-space) | Typical; not quantified in intrathecal T1 papers | Cardiac/respiratory gating or retrospective (finger PPG, bellows) |
| Tracer moves during long scans (Look-Locker 12 min, Mixed 8 min, VFA 17-24 min) | Not quantified; scans times are small relative to hours-long kinetics | |

Not found: any quantitative study of CSF flow bias on spoiled-GRE T1 mapping or on Look-Locker T1 in CSF. Ringstad/Eide T1w acquisitions are 3D GRE with FA 8 deg and the CSF signal there is not flow-compensated (not discussed).

---

## 7. Open gaps relevant to our design

1. No published CSF r2 for gadobutrol, nor r1 above ~1 mM; Behndig range was 0-1 mM, so injection-site (> 3 mM) R1/R2 and nonlinear behaviour are untested.
2. No published R1 mapping that covers both 0.1-0.5 s (high Gd) and 4.5 s in one acquisition; the field uses a two-sequence hybrid with threshold 1500 ms.
3. No stated accuracy requirement; the nearest are the +/-3 % tissue T1 margin and ~5 % Look-Locker reliability, and Look-Locker simulated error ~22 % at CSF T1 (Gonzo).
4. Full text of Valnes 2026 MRM (optimised Look-Locker; phantom r1/T1 repeatability) and Rohrer 2005 3 T table could not be read; worth obtaining.
5. Dura, cranial nerves, perivascular spaces: no quantitative T1/C values found (only the 2026 Fluids Barriers CNS dura-transport paper, doi:10.1186/s12987-026-00861-y, which uses the hybrid maps; content not read).

---

## References (URLs)

- Valnes et al. MRM 2026 https://onlinelibrary.wiley.com/doi/10.1002/mrm.70203 ; https://pubmed.ncbi.nlm.nih.gov/41340191/
- Behndig et al. MRM 2026 https://pmc.ncbi.nlm.nih.gov/articles/PMC13049235/ ; https://pubmed.ncbi.nlm.nih.gov/41621851/
- Ringstad 2018 https://insight.jci.org/articles/view/121537
- Watts 2019 https://www.ajnr.org/content/40/4/648 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC7048502/
- Eide 2023 dose https://pmc.ncbi.nlm.nih.gov/articles/PMC10171383/ ; Edeklev 2019 https://www.ajnr.org/content/40/8/1257
- Retention T1 https://pmc.ncbi.nlm.nih.gov/articles/PMC10425514
- Gonzo dataset https://pmc.ncbi.nlm.nih.gov/articles/PMC12909910/
- Storas https://www.medrxiv.org/content/10.1101/2025.07.19.25331835v1.full ; doi:10.1002/jmri.70169
- Hornkjol 2022 https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.932469/full
- Lee 2018 https://pmc.ncbi.nlm.nih.gov/articles/PMC5736474/ ; mouse https://pmc.ncbi.nlm.nih.gov/articles/PMC7471332/
- MP2RAGE CSF https://pmc.ncbi.nlm.nih.gov/articles/PMC13746071 ; Marques 2010 https://www.researchgate.net/publication/26885103
- MRF dictionary https://pmc.ncbi.nlm.nih.gov/articles/PMC10017611/
- Nataraj min-max https://pmc.ncbi.nlm.nih.gov/articles/PMC5319767/
- Dietrich https://pubmed.ncbi.nlm.nih.gov/24639175/
- Look-Locker incomplete recovery https://pmc.ncbi.nlm.nih.gov/articles/PMC3343218/
- Pintaske 2006 doi:10.1097/01.rli.0000197668.44926.f7 ; Szomolanyi 2019 https://pubmed.ncbi.nlm.nih.gov/31124800/ ; Rohrer 2005 doi:10.1097/01.RLI.0000184756.66360.D3
- Hagberg & Scheffler 2013 https://onlinelibrary.wiley.com/doi/full/10.1002/cmmi.1565
- CSF FLAIR artefacts https://pmc.ncbi.nlm.nih.gov/articles/PMC7973954/
