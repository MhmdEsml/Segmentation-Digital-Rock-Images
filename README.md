# Segmentation-Digital-Rock-Images
Welcome to the official GitHub repository for semantic segmentation of 11 digital rock images from the Digital Rock Portal.

## Installing the Requirements and Running the Inference Script
**1. Setting Up Environment**

Before running the inference script, ensure you have the required dependencies installed. You can install them using the following command:

<div id="codeSnippet">
  <pre><code>pip install -r requirements.txt</code></pre>
  <button onclick="copyCode('pip install -r requirements.txt')"></button>
</div>

This command will install all the necessary Python packages listed in the requirements.txt file.

**2. Running Dataset_Download.py**

Once you have installed the dependencies you can download and prepare the dataset using the following command:

<div id="codeSnippet">
  <pre><code>python Dataset_Download.py</code></pre>
  <button onclick="copyCode('python Dataset_Download.py')"></button>
</div>

After running this script, you can choose the dataset, the numbers of training, validation, and test data.

These datasets are available:
- Berea
- Bandera Brown
- Bandera Gray
- Bentheimer
- Berea Sister Gray
- Berea Upper Gray
- Buff Berea
- CastleGate
- Kirby
- Leopard
- Parker

Example:
- Please enter the image you want to download (e.g., Berea): `Berea`
- Please enter the number of train patches: 1000
- Please enter the number of validation patches: 100
- Please enter the number of test patches: 100

**3. Accessing Generated Images**

After running the inference script, you can find the generated images in the "./Generated_images" directory. Additionally, the script will compress the images into a zip file named Generated_images.zip for easier distribution and storage.

That's it! You have successfully run the inference script to generate images using the diffusion model. Feel free to explore and analyze the generated images for your digital rock analysis needs.

## Examples of Real and Generated Images

<table align="center">
  <tr>
    <td style="text-align: center;">
      <div>
        <img src="Images/8.png" alt="Sandstone Images">
        <figcaption>Sandstone Images</figcaption>
      </div>
    </td>
    <td style="text-align: center;">
      <div>
        <img src="Images/9.png" alt="Carbonate Images">
        <figcaption>Carbonate Images</figcaption>
      </div>
    </td>
  </tr>
</table>

## Citation

<div id="citation">
  <pre><code>@article{ESMAEILI2024127676,
  title = {Enhancing digital rock analysis through generative artificial intelligence: Diffusion models},
  journal = {Neurocomputing},
  volume = {587},
  pages = {127676},
  year = {2024},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2024.127676},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224004478},
  author = {Mohammad Esmaeili},
}</code></pre>
  <button onclick="copyCitation()"></button>
</div>
