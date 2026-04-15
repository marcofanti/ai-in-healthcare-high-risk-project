# Dataset Analysis: Data Types, Metadata, and Clinical Applications

## 1. `PKG - HistologyHSI-GB` (Hyperspectral Histological Images for Diagnosis of Human Glioblastoma)

**Dataset Size & Technical Details:**

- **Subjects & Scale:** 13 human subjects; 582 GB total dataset size.
- **Data Volume:** 482 total images (469 annotated hyperspectral images from 13 histological slides).
- **Imaging Specifications:** Exceptionally high-dimensional cubes. Each image has spatial dimensions of 800 × 1004 pixels and a spectral dimension of 826 channels across the visual and near-infrared range (400 to 1000 nm).
- **Acquisition:** H&E stained slides were captured using a custom push-broom hyperspectral system based on an Olympus BX-53 microscope at 20x magnification. Ground-truth annotations are binary (tumor "T" vs. non-tumor "NT").
- **Citation:** DOI: 10.7937/z1k6-vd17

**Data/File Types Found:**

- `.hdr` (ASCII Header file for the ENVI hyperspectral image format)
- `raw` data files without extensions (e.g., flat-binary raster files containing the multidimensional cubes, `whiteReference`, `darkReference`)
- `.png` (Standard RGB image previews)

**Pathology vs. Radiology Application:**
**Pathology.** This dataset contains Hyperspectral Imaging (HSI) scans of histological slides (Glioblastoma tissue). The ENVI format (.hdr + raw binary) captures a dense spectrum of light at each pixel. Because it operates at the microscopic tissue level, this is firmly a **Pathology** dataset used to answer medical questions regarding tissue classification (e.g., differentiating healthy brain tissue, necrosis, and active tumor margins based on distinct spectral signatures). We will rely on Python's `spectral` library to parse and load these files.

---

## 2. `IQ-OTH:NCCD - Lung Cancer Dataset`

**Dataset Size & Technical Details:**

- **Source:** Originally sourced from the Iraq-Oncology Teaching Hospital. Hosted and accessed via Kaggle.
- **Application Scope:** Used widely for screening abnormalities and computer-aided diagnostics focused on the pulmonary system.

**Data/File Types Found:**

- `.jpg` (JPEG image format)
- `.png` (PNG image format)

**Pathology vs. Radiology Application:**
**Radiology.** This dataset contains standard 2D image slices (JPEG and PNG) of Computed Tomography (CT) scans of the chest. Because CT scans are macroscopic images dealing with the internal structures of the body in cross-section, this dataset clearly belongs to **Radiology**. The images can be used to answer medical questions related to lung cancer, such as detecting pulmonary nodules, classifying them as benign vs. malignant, or screening for abnormalities.

---

## 3. `Oasis1` (Open Access Series of Imaging Studies: Cross-Sectional MRI)

**Dataset Size & Technical Details:**

- **Subjects & Demographics:** A cross-sectional collection of 416 subjects aged 18 to 96 (including both men and women, all right-handed). Includes 100 subjects over the age of 60 who have been clinically diagnosed with very mild to moderate Alzheimer's disease (AD).
- **Reliability Cohort:** Also includes a reliability dataset of 20 non-demented subjects scanned on a subsequent visit within 90 days of their initial session.
- **Acquisition:** For each subject, 3 or 4 individual T1-weighted structural MRI scans obtained in single scan sessions are included. The image data is approximately 18 GB in size (distributed across 12 `.tar.gz` archives).
- **Source & Citation:** Washington University Alzheimer's Disease Research Center. (DOI: 10.1162/jocn.2007.19.9.1498)

**Data/File Types Found:**

- `.img` and `.hdr` (Analyze 7.5 format pairs, representing the raw and processed 3D MRI volumes)
- `.gif` (2D preview thumbnails of the MRI slices)
- `.csv` (Tabular files containing demographic and clinical metadata, such as Clinical Dementia Rating (CDR), Mini-Mental State Examination (MMSE) scores, age, sex, education, and estimated total intracranial volume)

**Pathology vs. Radiology Application:**
**Radiology.** This dataset contains 3D structural brain MRI volumes. Like CT scans, MRI imaging operates on macroscopic bodily structures, placing this firmly within **Radiology**. The `TissueLab-SDK` NIfTI wrapper can typically adapt to Analyze formats, or standard neuroimaging libraries like `nibabel` will be heavily utilized to parse the 3D structures. The images and accompanying clinical `.csv` data can answer medical questions such as quantifying brain atrophy or aiding in Alzheimer's / dementia classification models based on whole-brain volume changes.

---

## 4. `Quilt-1M` (One Million Image-Text Pairs for Histopathology)

**Dataset Size & Technical Details:**

- **Size:** 1 million paired image-text samples, making it the largest open-source vision-language histopathology dataset to date.
- **Sources:** Automatically curated from 1,087 hours of YouTube educational histopathology videos, subsequently augmented with data from PubMed open-access articles, Twitter (OpenPath), and LAION.
- **Features:** Spans 18 distinct sub-pathology categories and captures imagery at varying microscopic magnification scales (e.g., 10x, 20x, 40x). Used to train multi-modal foundation models like QuiltNet.
- **Publication:** Featured at NeurIPS 2023 (Oral).

**Data/File Types Found:**

- `.jpg` (~718,000 image files)
- `.png` (~3,000 image files)
- **Rich Text Metadata:** Detailed text annotations containing explicitly separated variables such as _Medical text_, _Region-of-Interest (ROI) text_, extracted UMLS entities, and localized sub-pathology classifications.

**Pathology vs. Radiology Application:**
**Pathology.** This is an enormous dataset of histopathology microscopy images (typically H&E stained tissue slides). Because it consists of cell- and tissue-level micrographs captured through optical scopes, it is unquestionably a **Pathology** dataset. It answers foundational pathological questions such as cancer cell grading, tissue recognition, and serves as a benchmark for multi-modal tasks like visual question answering across dozens of sub-specialties. Standard image wrappers (e.g., `SimpleImageWrapper` in `TissueLab-SDK`) can be used to load these files.

---

## 5. `Spinal` (Spinal-Multiple-Myeloma-SEG)

**Dataset Size & Technical Details:**

- **Subjects & Scale:** 67 patients spanning 72 scans; 304.43 GB total dataset size. Features 576 image series comprising 564,464 axial slices.
- **Imaging Modalities:** Acquired via dual-layer dual-energy CT (Philips IQon Spectral CT). Features conventional CT images alongside advanced parametric reconstructions: Virtual Monoenergetic Images (VMI at 40, 80, and 120 keV) and Calcium-suppressed images.
- **Annotations:** Detailed lesion and vertebrae mask segmentations (including vertebra type classification) generated using nnU-Net v2 and subjected to iterative manual refinement by expert radiologists.
- **Citation:** DOI: 10.7937/k4qv-hh78

**Data/File Types Found:**

- `.dcm` (DICOM format containing the raw full 3D Spectral CT slices)
- `.nii.gz` and `DICOM-SEG` (Compressed NIfTI and DICOM Segmentation formats housing the 3D annotation masks)
- `.tsv` (Tab-Separated Values containing rich clinical context, basic demographics, disease progression milestones, treatment regimens, and molecular test results)

**Pathology vs. Radiology Application:**
**Radiology.** This dataset houses the raw, rigorous files behind spinal spectral CT scans. This includes hundreds of thousands of standard medical DICOM slices from a clinical scanner, combined with highly processed `.nii.gz` and `DICOM-SEG` annotation masks. Because this targets deep internal anatomy (the human spine), it belongs in **Radiology**. This imagery answers questions surrounding multiple myeloma lesion detection, segmentation, and tumor burden assessment by mapping voxel densities. The `TissueLab-SDK` `DicomImageWrapper` and `NiftiImageWrapper` will be extremely relevant here, alongside tools to parse the accompanying `.tsv` tables.

---

## 6. Model Compatibility: `CheXagent-2-3b` (Chest X-ray Foundation Model)

**Model Overview:**
`CheXagent-2-3b` is an instruction-tuned foundation model specifically designed for analyzing Chest X-ray (CXR) images. It is built on the InternVL2 architecture (~3B parameters) and is optimized for radiology tasks such as finding generation, disease identification, and anatomical description.

**Compatibility with Project Datasets:**

| Dataset | Compatibility | Rationale |
| :--- | :--- | :--- |
| **IQ-OTH:NCCD** | **High** | Contains Chest CT slices. While the model is trained on X-rays (projection), Chest CT (cross-section) shares the same anatomical domain and pathological features (lung nodules, masses). |
| **Spinal** | **Medium** | Contains Spinal CT scans. Although the primary domain is different (Spine vs. Chest), the model's general radiology knowledge can assist in identifying bone structures and density abnormalities in CT slices. |
| **Oasis1** | **Low** | Brain MRI. Significant domain shift (different modality and anatomy). Included primarily for testing model robustness across modalities. |
| **Quilt-1M** | **None** | Pathology (microscopy). Out-of-domain. The model is not designed for cell-level tissue analysis. |
| **PKG HSI** | **None** | Pathology (hyperspectral). Out-of-domain. Requires specialized spectral parsing and tissue-level analysis. |

**Implementation in `week2_gemini/test_CheXagent-8b.ipynb`:**
Local samples from `IQ-OTH:NCCD` and `Spinal` are analyzed by extracting 2D slices and passing them to the model with standard clinical prompts. Non-standard formats (DICOM) are automatically converted to temporary JPEGs for inference.

