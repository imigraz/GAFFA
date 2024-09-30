# GAFFA

![GAFFA Graphical Abstract](graphical_abstract.png)

This is the official code repository for the paper:  

**Implicit Is Not Enough: Explicitly Enforcing Anatomical Priors inside Landmark Localization Models**

[https://doi.org/10.3390/bioengineering11090932](https://doi.org/10.3390/bioengineering11090932)

## 1. Installation

1. Move dataset images to `datasets/xray_hand/images`. Annotations are already included in the repository.

2. Setup a conda environment (e.g., with miniconda) with Python 3.10.14:

   ```bash
   conda create -n GAFFA python=3.10.14

3. Install all needed Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## 2. Usage

We provided some example configuration files that can be used with:

```bash
python main.py default/default_xray_hand_train_UNet+GAFFA.json
```

**Note:** Code will be uploaded soon.
